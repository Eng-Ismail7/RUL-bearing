from sklearn.svm import LinearSVR

def fit_svr(X, y, kernel: str = 'rbf') -> LinearSVR:
    """
    Fit support vector regression for the given input X and expected labes y.
    :param X: Feature data
    :param y: Labels that should be correctly computed
    :param kernel: type of kernel used by the SVR {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    :return: SVR that is fitted to X and y
    """
    svr = LinearSVR()
    svr.fit(X=X, y=y)
    return svr
