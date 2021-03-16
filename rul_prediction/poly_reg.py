from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def fit_poly_reg(X, y, degree=1, memory_path=None) -> Pipeline:
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression(), memory=memory_path)
    polyreg.fit(X, y)
    return polyreg
