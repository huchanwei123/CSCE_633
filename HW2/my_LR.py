import numpy as np
import math

class LinearRegression(object):
    def __init__(self, reg=False, _lambda=0.1):
        self.reg = reg
        self._lambda = _lambda

    def fit(self, X_train, y_train):
        # compute peusdo inverse
        XTX = np.matmul(np.transpose(X_train), X_train)
        if self.reg:
            s = int(np.shape(XTX)[0])
            lambda_mat = self._lambda * np.eye(s)
            X_inv = np.matmul(np.linalg.pinv(XTX + lambda_mat), \
                          np.transpose(X_train))
        else:
            X_inv = np.matmul(np.linalg.pinv(XTX), \
                          np.transpose(X_train))
        self.weight = np.matmul(X_inv, y_train)

        return self.weight

    def predict(self, X_test):
        # predict the output
        self.pred = np.matmul(X_test, self.weight)
        #print('Predict output is: {}'.format(self.pred))
        return self.pred

    def RSS_error(self, y_test):
        RSS = sum(np.power(self.pred - y_test, 2))
        #print('True output is: {}'.format(y_test))
        #print('------------------------------------')
        return math.sqrt(RSS)
