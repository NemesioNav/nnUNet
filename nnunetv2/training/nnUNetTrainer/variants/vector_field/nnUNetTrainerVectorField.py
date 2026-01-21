"""
Custom trainer for vector field segmentation data.

This trainer handles datasets where the ground truth contains:
- Channel 0: Binary segmentation mask
- Channels 1-3: XYZ vector components (direction vectors)

Key modifications:
1. Uses VectorFieldDataLoader (float32 segmentation)
2. Custom loss: Dice+CE for mask, MSE for vectors
3. Network outputs 4 channels: 1 mask (sigmoid) + 3 vectors
4. No spatial data augmentation - vectors break with rotation/mirroring

Available trainers:
- nnUNetTrainerVectorFieldNoDA: No augmentation at all
- nnUNetTrainerVectorField: Safe intensity augmentations only (no spatial)
"""

import multiprocessing
import warnings
from time import sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import distributed as dist

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.vector_field_data_loader import VectorFieldDataLoader
from nnunetv2.training.loss.compound_losses import DC_BCE_and_MSE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
# from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
# from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
# from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainerVectorFieldNoDA(nnUNetTrainer):
    """
    Trainer for vector field data without data augmentation.

    Use this for the first experiments. Spatial augmentations (rotation, mirroring)
    would require special handling of vector components.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # We output 4 channels: 1 for mask + 3 for vectors
        self.num_vector_channels = 3

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """No data augmentation - use validation transforms for training."""
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Disable mirroring for inference as well."""
        rotation_for_DA, do_dummy_2d_data_aug, _, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        initial_patch_size = self.configuration_manager.patch_size
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def initialize(self):
        """Override to use custom number of output channels."""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                    self.dataset_json)

            # Custom: output channels = 1 (mask) + 3 (vectors) = 4
            # Using 1 for mask because we'll use sigmoid, not softmax
            num_output_channels = 1 + self.num_vector_channels

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                num_output_channels,
                self.enable_deep_supervision
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # Empty CUDA cache to reclaim memory after network initialization
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        """Build vector field loss: Dice+BCE for mask, MSE for vectors."""
        loss = DC_BCE_and_MSE_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 1e-5, 'do_bg': True, 'ddp': self.is_ddp},
            bce_kwargs={},
            weight_dice=1.0,
            weight_bce=1.0,
            weight_mse=1.0,
            apply_mask_to_vectors=True,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def get_dataloaders(self):
        """Override to use VectorFieldDataLoader."""
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Use VectorFieldDataLoader instead of nnUNetDataLoader
        dl_tr = VectorFieldDataLoader(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = VectorFieldDataLoader(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling)

        # No augmenter wrapping needed since we're using no DA
        return dl_tr, dl_val

        # allowed_num_processes = get_allowed_n_proc_DA()
        # if allowed_num_processes == 0:
        #     mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        #     mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        # else:
        #     mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
        #                                                 num_processes=allowed_num_processes,
        #                                                 num_cached=max(6, allowed_num_processes // 2), seeds=None,
        #                                                 pin_memory=self.device.type == 'cuda', wait_time=0.002)
        #     mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
        #                                               transform=None, num_processes=max(1, allowed_num_processes // 2),
        #                                               num_cached=max(3, allowed_num_processes // 4), seeds=None,
        #                                               pin_memory=self.device.type == 'cuda',
        #                                               wait_time=0.002)

        # _ = next(mt_gen_train)
        # _ = next(mt_gen_val)
        # return mt_gen_train, mt_gen_val

    def validation_step(self, batch: dict) -> dict:
        """Validation step - compute loss and mask Dice."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        # For validation metrics, we only look at the mask (channel 0)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Get mask predictions (sigmoid > 0.5)
        pred_mask = torch.sigmoid(output[:, 0:1]) > 0.5
        target_mask = target[:, 0:1] > 0.5

        # Compute Dice for mask
        axes = [0] + list(range(2, output.ndim))
        tp = (pred_mask & target_mask).sum(axes)
        fp = (pred_mask & ~target_mask).sum(axes)
        fn = (~pred_mask & target_mask).sum(axes)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        return {'loss': loss.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Validation for vector field data.

        Saves raw logits for all 4 channels (mask + xyz vectors) as .npz files.
        Does not compute standard segmentation metrics since ground truth has
        multiple channels (mask + vectors).
        """
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as export_pool:
            worker_list = [i for i in export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]  # convert blosc2 to numpy

                if self.is_cascaded:
                    raise NotImplementedError("Cascade training is not supported for vector field data")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # Export raw logits for vector field data
                results.append(
                    export_pool.starmap_async(
                        export_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             output_filename_truncated, default_num_processes),
                        )
                    )
                )

                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file(f"Predictions saved to {validation_output_folder}", also_print_to_console=True)
            self.print_to_log_file("Note: Standard Dice metrics not computed for vector field data.",
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


class nnUNetTrainerVectorField(nnUNetTrainerVectorFieldNoDA):
    """
    Trainer for vector field data with SAFE intensity augmentations.

    Includes intensity-based augmentations that don't affect vector directions:
    - Gaussian noise
    - Gaussian blur
    - Brightness
    - Contrast
    - Gamma
    - Simulate low resolution

    Excludes spatial augmentations that would break vectors:
    - Rotation
    - Mirroring
    - Transpose
    - Elastic deformation
    """

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        Training transforms with safe intensity augmentations only.
        No spatial augmentations (rotation, mirroring, etc.) that would break vectors.
        """
        transforms = []

        # === SAFE INTENSITY AUGMENTATIONS ===
        # (No spatial transforms - they would break vector directions)

        # Gaussian Noise
        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ),
                apply_probability=0.1
            )
        )

        # Gaussian Blur
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2
            )
        )

        # Brightness
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                apply_probability=0.15
            )
        )

        # Contrast
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                apply_probability=0.15
            )
        )

        # Simulate Low Resolution
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=None,
                    allowed_channels=None,
                    p_per_channel=0.5
                ),
                apply_probability=0.25
            )
        )

        # Gamma Transform (two variants like in default nnUNet)
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                apply_probability=0.1
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                apply_probability=0.3
            )
        )

        # === END INTENSITY AUGMENTATIONS ===

        # Mask normalization if needed (before RemoveLabelTransform)
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0
                )
            )

        # Remove label transform (after augmentations, before region conversion)
        transforms.append(RemoveLabelTansform(-1, 0))

        # Note: Cascade not supported for vector field data
        if is_cascaded:
            raise NotImplementedError("Cascade training is not supported for vector field data")

        # Region conversion if needed
        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # Deep supervision downsampling
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerVectorField100epochs(nnUNetTrainerVectorField):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerVectorFieldNoDA100epochs(nnUNetTrainerVectorFieldNoDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
