"""
Mean-Variance-Standard Deviation Calculator

Create a function named calculate() in mean_var_std.py that uses Numpy to output the mean, variance, standard deviation, max, min, and sum of the rows, columns, and elements in a 3 x 3 matrix.

The input of the function should be a list containing 9 digits. The function should convert the list into a 3 x 3 Numpy array, and then return a dictionary containing the mean, variance, standard deviation, max, min, and sum along both axes and for the flattened matrix.

The returned dictionary should follow this format:

{
  'mean': [axis1, axis2, flattened],
  'variance': [axis1, axis2, flattened],
  'standard deviation': [axis1, axis2, flattened],
  'max': [axis1, axis2, flattened],
  'min': [axis1, axis2, flattened],
  'sum': [axis1, axis2, flattened]
}
If a list containing less than 9 elements is passed into the function, it should raise a ValueError exception with the message: "List must contain nine numbers." The values in the returned dictionary should be lists and not Numpy arrays.

For example, calculate([0,1,2,3,4,5,6,7,8]) should return:

{
  'mean': [[3.0, 4.0, 5.0], [1.0, 4.0, 7.0], 4.0],
  'variance': [[6.0, 6.0, 6.0], [0.6666666666666666, 0.6666666666666666, 0.6666666666666666], 6.666666666666667],
  'standard deviation': [[2.449489742783178, 2.449489742783178, 2.449489742783178], [0.816496580927726, 0.816496580927726, 0.816496580927726], 2.581988897471611],
  'max': [[6, 7, 8], [2, 5, 8], 8],
  'min': [[0, 1, 2], [0, 3, 6], 0],
  'sum': [[9, 12, 15], [3, 12, 21], 36]
}
"""

import numpy as np


def calculate(list_numbers):
    """
    Calculates various statistics (mean, variance, standard deviation, maximum, minimum, and sum) of a 3x3 matrix.

    Args:
        list_numbers (list): A list of nine numbers.

    Returns:
        dict: A dictionary containing the mean, variance, standard deviation, maximum, minimum, and sum across various dimensions of the input list of numbers.
    """

    # check if the list contains nine numbers
    if len(list_numbers) < 9:
        raise ValueError("List must contain nine numbers.")

    # convert list to 3x3 numpy array
    numpy_array = np.array(list_numbers).reshape((3, 3))

    # calculate mean along different dimensions
    mean_1, mean_2, mean_flat = (
        numpy_array.mean(axis=0),
        numpy_array.mean(axis=1),
        numpy_array.mean(),
    )

    # calculate variance along different dimensions
    var_1, var_2, var_flat = (
        numpy_array.var(axis=0),
        numpy_array.var(axis=1),
        numpy_array.var(),
    )

    # calculate standard deviation along different dimensions
    std_1, std_2, std_flat = (
        numpy_array.std(axis=0),
        numpy_array.std(axis=1),
        numpy_array.std(),
    )

    # calculate maximum along different dimensions
    max_1, max_2, max_flat = (
        numpy_array.max(axis=0),
        numpy_array.max(axis=1),
        numpy_array.max(),
    )

    # calculate minimum along different dimensions
    min_1, min_2, min_flat = (
        numpy_array.min(axis=0),
        numpy_array.min(axis=1),
        numpy_array.min(),
    )

    # calculate sum along different dimensions
    sum_1, sum_2, sum_flat = (
        numpy_array.sum(axis=0),
        numpy_array.sum(axis=1),
        numpy_array.sum(),
    )

    # create a dictionary with calculated statistics
    calculations = {
        "mean": [[*mean_1], [*mean_2], mean_flat],
        "variance": [[*var_1], [*var_2], var_flat],
        "standard deviation": [[*std_1], [*std_2], std_flat],
        "max": [[*max_1], [*max_2], max_flat],
        "min": [[*min_1], [*min_2], min_flat],
        "sum": [[*sum_1], [*sum_2], sum_flat],
    }

    return calculations