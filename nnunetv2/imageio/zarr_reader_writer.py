#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
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

from typing import Tuple, Union, List
import numpy as np
import zarr

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class ZarrIO(BaseReaderWriter):
    supported_file_endings = [
        '.zarr',
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads zarr arrays from specified paths.
        
        For synthetic datasets without spacing information, spacing is set to 1.0 for all dimensions.
        For 2D images, spacing is set to (999, 1, 1) to ensure proper handling.
        
        :param image_fnames: Tuple/List of paths to .zarr directories (one per image channel)
        :return: Tuple of (stacked image array of shape (c, x, y, z), metadata dict)
        """
        images = []
        spacings_for_nnunet = []
        
        for f in image_fnames:
            # Open zarr array from the specified path
            zarr_array = zarr.open_array(f, mode='r')
            npy_image = np.array(zarr_array)
            
            if npy_image.ndim == 2:
                # 2D image: add channel and batch dimensions
                # Expected shape after: (1, 1, x, y)
                npy_image = npy_image[None, None]  # Add channel and depth dimensions
                spacings_for_nnunet.append((999, 1, 1))  # 999 for depth to indicate 2D
            elif npy_image.ndim == 3:
                # 3D image: add channel dimension
                # Expected shape after: (1, x, y, z)
                npy_image = npy_image[None]  # Add channel dimension
                spacings_for_nnunet.append((1, 1, 1))  # Default spacing of 1 for synthetic data
            elif npy_image.ndim == 4:
                # 4D image: already has channels
                # Expected shape: (c, x, y, z)
                spacings_for_nnunet.append((1, 1, 1))
            else:
                raise RuntimeError(
                    f"Unsupported image dimensions: {npy_image.ndim}. "
                    f"Expected 2D, 3D, or 4D arrays. File: {f}"
                )
            
            images.append(npy_image)
        
        # Validate all images have the same shape
        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        
        # Validate all images have the same spacing
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        
        # Stack all channels and return with metadata
        stacked_images = np.vstack(images, dtype=np.float32, casting='unsafe')
        
        properties_dict = {
            'spacing': tuple(spacings_for_nnunet[0])
        }
        
        return stacked_images, properties_dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Reads segmentation from a zarr array.
        
        Segmentations should be stored as 3D arrays (or 2D for 2D data).
        The output will be reshaped to (1, x, y, z) or (1, 1, x, y) for 2D.
        
        TEMPORARY FIX: If segmentation has 4 channels, only keep the first (labels),
        discard the last 3 (xyz vectors).
        
        :param seg_fname: Path to the segmentation .zarr directory
        :return: Tuple of (segmentation array of shape (1, x, y, z), metadata dict)
        """
        # Use read_images to load the segmentation
        seg_array, properties = self.read_images((seg_fname,))
        
        # TEMPORARY FIX: Handle 4-channel segmentation (label + xyz vectors)
        if seg_array.shape[0] == 4:
            # Keep only the first channel (segmentation labels), discard xyz vectors
            seg_array = seg_array[0:1]
            # print(f"NOTE: Segmentation had 4 channels. Keeping only first channel (labels), "
            #       f"discarding last 3 channels (xyz vectors). New shape: {seg_array.shape}")

        # TEMPORARY FIX: Handle non-consecutive labels by relabeling to consecutive integers
        unique_labels = np.unique(seg_array)
        if not np.array_equal(unique_labels, np.arange(unique_labels.min(), unique_labels.max() + 1)):
            new_seg = np.zeros_like(seg_array)
            for new_label, old_label in enumerate(unique_labels):
                new_seg[seg_array == old_label] = new_label
            seg_array = new_seg
            # print(f"NOTE: Segmentation labels were not consecutive. Relabeled to consecutive integers. "
            #       f"New unique labels: {np.unique(seg_array)}")

        return seg_array, properties

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Writes segmentation to a zarr array.
        
        :param seg: Segmentation array of shape (1, x, y, z) or (1, 1, x, y) for 2D
        :param output_fname: Output path for the .zarr directory
        :param properties: Metadata dictionary (not used for zarr, but required by interface)
        """
        # Remove the channel dimension for zarr storage
        assert seg.ndim == 3 or seg.ndim == 4, \
            f'Segmentation must be 3D (1, x, y, z) or 4D (1, 1, x, y) for 2D. Got shape {seg.shape}'
        
        if seg.ndim == 4:
            # 2D case: shape is (1, 1, x, y), remove both dimensions
            seg_to_save = seg[0, 0]
        else:
            # 3D case: shape is (1, x, y, z), remove channel dimension
            seg_to_save = seg[0]
        
        # Convert to appropriate integer dtype
        seg_to_save = seg_to_save.astype(
            np.uint8 if np.max(seg_to_save) < 255 else np.uint16,
            copy=False
        )
        
        # Create and write zarr array
        zarr_array = zarr.open_array(output_fname, mode='w', shape=seg_to_save.shape, 
                                     dtype=seg_to_save.dtype, chunks=None)
        zarr_array[:] = seg_to_save