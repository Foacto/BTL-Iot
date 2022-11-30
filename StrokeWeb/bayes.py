import numpy as np

class NaiveBayes:
    #X - training data
    #y - training label
    def fit(self, X, y):
        #Xác định sô lương hàng và côt
        n_samples, n_features = X.shape
        #Xác định các giá trị của nhãn
        self._classes = np.unique(y)
        #Xác định sô lương nhãn trong tâp dư liêu
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        #Giá trị trung bình của môĩ fearture ưng voi môi nhãn
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            #get the sample has this class
            X_c = X[y == c]
            a = self._mean[idx, :] = X_c.mean(axis=0)
            b = self._var[idx, :] = X_c.var(axis=0)
            #Xác định tân suât của nhãn đó trong tâp dư liêu
            #X_c.shape[0] sô lương mâũ của nhãn
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    #X - more than 1 samples
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
