from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from math import sqrt
from numpy import std

from util.constants import RMSE_KEY, CORR_COEFF_KEY, STANDARD_DEVIATION_KEY


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


@rename(RMSE_KEY)
def rmse(y_actual, y_predicted):
    """
    Calculates sqrt(1/n * sum((y_actual-y_predicted)^2))
    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: root mean square
    """
    return sqrt(mean_squared_error(y_actual, y_predicted))


def mae(y_actual, y_predicted):
    """
    Calculates 1/n * sum(|y_actual-y_predicted|)
    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: root mean square
    """
    return mean_absolute_error(y_actual, y_predicted)


@rename(CORR_COEFF_KEY)
def correlation_coefficient(y_actual, y_predicted):
    coeff, _ = pearsonr(y_actual, y_predicted)
    return coeff


@rename(STANDARD_DEVIATION_KEY)
def standard_deviation(value_list):
    return std(value_list)
