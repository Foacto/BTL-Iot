import numpy as np

class NaiveBayes:
    #X - dư liêụ train
    #y - dư liêụ test
    def fit(self, X, y):
        #Xác định sô lương mâũ và đăc trưng của data 
        sl_mau, sl_dactrung = X.shape
        #Xác định các giá trị của nhãn
        self.nhan = np.unique(y)
        #Xác định sô lương nhãn trong tâp dư liêu
        sl_nhan = len(self.nhan)

        #Giá trị trung bình của môĩ fearture ưng voi môi nhãn
        self.trungbinh = np.zeros((sl_nhan, sl_dactrung), dtype=np.float64)
        self.phuongsai = np.zeros((sl_nhan, sl_dactrung), dtype=np.float64)
        self._priors = np.zeros(sl_nhan, dtype=np.float64)

        for chiso, giatri_nhan in enumerate(self.nhan):
            #Lâý các mâũ có nhãn băng vơi giá trị của tưng nhãn
            X_c = X[y == giatri_nhan]
            self.trungbinh[chiso, :] = X_c.mean(axis=0)
            self.phuongsai[chiso, :] = X_c.var(axis=0)
            #Xác định tân suât của nhãn đó trong tâp dư liêu
            #X_c.shape[0] sô lương mâũ của nhãn
            self._priors[chiso] = X_c.shape[0] / float(sl_mau)
    
    #X - more than 1 samples
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self.nhan):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x))) + prior
            posteriors.append(posterior)
        # return class with highest posterior probability
        return self.nhan[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # print(x)
        gt_trungbinh_nhan = self.trungbinh[class_idx]
        gt_phuongsai_nhan = self.phuongsai[class_idx]
        tuso = np.exp(-((x - gt_trungbinh_nhan) ** 2) / (2 * gt_phuongsai_nhan))
        mauso = np.sqrt(2 * np.pi * gt_phuongsai_nhan)
        return tuso / mauso
