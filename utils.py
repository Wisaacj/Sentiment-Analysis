import numpy as np
import pandas as pd
import numpy.typing as npt


def convert_to_nd_array(series: pd.Series) -> npt.NDArray:
    """
    Converts a Pandas Series to a vertically-stacked NumPy array.

    :param series: the Pandas Series to convert.
    :returns: a vertically-stacked NumPy array.
    """
    return np.vstack(series.apply(np.array))