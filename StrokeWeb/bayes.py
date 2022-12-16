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
        self.tansuat_nhan = np.zeros(sl_nhan, dtype=np.float64)

        for chiso, giatri_nhan in enumerate(self.nhan):
            #Lâý các mâũ có nhãn băng vơi giá trị của tưng nhãn
            X_c = X[y == giatri_nhan]
            self.trungbinh[chiso, :] = X_c.mean(axis=0)
            self.phuongsai[chiso, :] = X_c.var(axis=0)
            #Xác định tân suât của nhãn đó trong tâp dư liêu
            #X_c.shape[0] sô lương mâũ của nhãn
            self.tansuat_nhan[chiso] = X_c.shape[0] / float(sl_mau)
    
    def predict(self, X):
        dudoan = [self._predict(x) for x in X]
        return np.array(dudoan)

    def _predict(self, x):
        mang_xacsuat_nhan = []
        # Tính xăc suât hâụ nghiêm cho các nhãn
        for idx, c in enumerate(self.nhan):
            xacsuat_nhan = np.sum(np.log(self._pdf(idx, x))) + np.log(self.tansuat_nhan[idx])
            mang_xacsuat_nhan.append(xacsuat_nhan)
        # Trả vê nhãn vơi giá trị xs hâụ nghiêm max
        return self.nhan[np.argmax(mang_xacsuat_nhan)]

    def _pdf(self, class_idx, x):
        # print(x)
        gt_trungbinh_nhan = self.trungbinh[class_idx]
        gt_phuongsai_nhan = self.phuongsai[class_idx]
        tuso = np.exp(-((x - gt_trungbinh_nhan) ** 2) / (2 * gt_phuongsai_nhan))
        mauso = np.sqrt(2 * np.pi * gt_phuongsai_nhan)
        return tuso / mauso
