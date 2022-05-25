"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

from typing import List, Union
from matplotlib import image, scale
import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        def _scale_one(X: np.ndarray) -> np.ndarray:
            X_std = (X - self._data_min) / (self._data_max - self._data_min)
            X_scaled = X_std * (self.max - self.min) + self.min
            return X_scaled

        if images is list:
            return [_scale_one(i) for i in images]
        return _scale_one(images)


def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean_arr, std_arr = [], []
    for idx in range(images.shape[-1]):
        # We unwrap all channel values in a Nx1 vector
        ch_stack = images[:, :, :, idx].reshape(-1)
        mean_arr.append(np.mean(ch_stack))
        std_arr.append(np.std(ch_stack))
    if len(mean_arr) == 1:
        return mean_arr[0], std_arr[0]
    return mean_arr, std_arr

class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
