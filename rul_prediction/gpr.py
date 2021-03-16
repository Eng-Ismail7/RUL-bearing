from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from util.constants import MEMORY_CACHE_PATH
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def fit_gpr(X, y) -> GaussianProcessRegressor:
    gpr = GaussianProcessRegressor(random_state=0)
    gpr.fit(X=X, y=y)
    
    return gpr
