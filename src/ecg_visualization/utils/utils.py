from typing import Tuple
from numpy.typing import NDArray
import math
import numpy as np


def padding_reshape(array: NDArray, shape: Tuple, fill_value=np.nan):
    total = math.prod(shape)

    if len(array) < total:
        pad_length = total - len(array)
        array = np.concatenate([array, np.full(pad_length, fill_value)])

    return np.reshape(array, shape)
