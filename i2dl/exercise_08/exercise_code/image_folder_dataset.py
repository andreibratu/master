"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import random
import numpy as np

import torch

from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""

    def __init__(self, *args,
                 root=None,
                 images=None,
                 labels=None,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/mnist.zip",
                 **kwargs):
        super().__init__(*args,
                         download_url=download_url,
                         root=root,
                         **kwargs)
        print(download_url)
        self.images = torch.load(os.path.join(root, images))
        if labels is not None:
            self.labels = torch.load(os.path.join(root, labels))
        else:
            self.labels = None
        # transform function that we will apply later for data preprocessing
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image

class RandomHorizontalFlip:
    """
    Transform class that flips an image horizontically randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        :param prob: Probability of the image being flipped
        """
        self.p = prob

    def __call__(self, image):
        rand = random.uniform(0,1)
        if rand < self.p:
            image = np.flip(image,1)
        return image