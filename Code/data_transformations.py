import torch
from torch import Tensor
from torch.nn import functional as F
import random

import rep_transformations as rt

class DctBlurry(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, threshold, prob=0.2):
        super().__init__()
        self.threshold = threshold
        self.prob = 1

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if self.prob > random.random():
            tensor_dct = rt.fdct(tensor)
            tensor_dct[:,self.threshold:,self.threshold:] = 0
            result = rt.idct(tensor_dct)
        else:
            result = tensor
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"