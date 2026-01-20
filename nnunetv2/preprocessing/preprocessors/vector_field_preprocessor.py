#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Custom preprocessor for vector field segmentation data.

This preprocessor handles datasets where the ground truth contains:
- Channel 0: Binary segmentation mask
- Channels 1-3: XYZ vector components (direction vectors)

The mask goes through the normal preprocessing pipeline, then the vectors
are read separately, cropped using the same bounding box, and concatenated
to create a 4-channel output stored as float32.
"""

from typing import List, Union
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.imageio.zarr_reader_writer import ZarrIO
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class VectorFieldPreprocessor(DefaultPreprocessor):
    """
    Preprocessor for vector field segmentation data.

    Extends DefaultPreprocessor to handle 4-channel ground truth:
    - Channel 0: Binary mask (processed normally, stored as float but with int values)
    - Channels 1-3: XYZ vector components (cropped and stored as float32)

    The vectors are not resampled, so this should only be used with 3d_fullres
    configuration where no resampling occurs (target spacing = original spacing).
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.zarr_reader = ZarrIO()

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        """
        Process a single case and save with vector field data.

        1. Process mask through normal pipeline (transpose, crop, resample, normalize)
        2. Read raw vectors and apply same transpose and crop (no resample)
        3. Concatenate mask + vectors into 4-channel tensor
        4. Save as float32
        """
        # Run normal preprocessing for mask
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager,
                                               configuration_manager, dataset_json)

        # seg is now shape (1, x, y, z) - the preprocessed mask
        # It has been: transposed, cropped, resampled (if needed), converted to int8/int16

        # Read raw vectors from the segmentation file
        vectors, _ = self.zarr_reader.read_vectors(seg_file)
        # vectors shape: (3, x, y, z) - raw, not preprocessed

        # Apply the same transpose as was applied to the mask
        transpose_forward = plans_manager.transpose_forward
        # Transpose spatial axes of vectors
        vectors = vectors.transpose([0, *[i + 1 for i in transpose_forward]])

        # For vector components, we also need to reorder the channels to match the new axis order
        # If axes were reordered, the vector components (vx, vy, vz) must also be reordered
        # to (v_new_axis0, v_new_axis1, v_new_axis2)
        vectors = vectors[transpose_forward]

        # Apply the same crop using the bounding box from properties
        bbox = properties['bbox_used_for_cropping']
        slicer = bounding_box_to_slice(bbox)
        slicer = (slice(None),) + slicer  # Add channel dimension
        vectors = vectors[slicer]

        # Verify shapes match
        if vectors.shape[1:] != seg.shape[1:]:
            raise RuntimeError(
                f"Shape mismatch after cropping! Mask shape: {seg.shape}, Vectors shape: {vectors.shape}. "
                f"This may indicate resampling occurred. VectorFieldPreprocessor should only be used "
                f"with 3d_fullres configuration where target spacing equals original spacing."
            )

        # Concatenate mask and vectors: (1, x, y, z) + (3, x, y, z) -> (4, x, y, z)
        # Convert mask to float32 (it was int8/int16)
        seg_with_vectors = np.concatenate([seg.astype(np.float32), vectors.astype(np.float32)], axis=0)

        if self.verbose:
            print(f"Combined seg shape: {seg_with_vectors.shape}, dtype: {seg_with_vectors.dtype}")
            print(f"  Mask channel - min: {seg_with_vectors[0].min()}, max: {seg_with_vectors[0].max()}")
            print(f"  Vector channels - min: {seg_with_vectors[1:].min()}, max: {seg_with_vectors[1:].max()}")

        # Save data and combined segmentation
        data = data.astype(np.float32, copy=False)
        # Keep seg as float32 (NOT int16!) to preserve vector values
        seg_with_vectors = seg_with_vectors.astype(np.float32, copy=False)

        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
            data.shape,
            tuple(configuration_manager.patch_size),
            data.itemsize)
        block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(
            seg_with_vectors.shape,
            tuple(configuration_manager.patch_size),
            seg_with_vectors.itemsize)

        nnUNetDatasetBlosc2.save_case(data, seg_with_vectors, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg)
