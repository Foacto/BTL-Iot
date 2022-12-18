import numpy as np


class NaiveBayes:
    # X - dư liêụ train
    # y - dư liêụ test
    def fit(self, X, y):
        # Xác định sô lương mâũ và đăc trưng của data
        sl_mau, sl_dactrung = X.shape
        # Xác định các giá trị của nhãn
        self.nhan = np.unique(y)
        # Xác định sô lương nhãn trong tâp dư liêu
        sl_nhan = len(self.nhan)

        # Giá trị trung bình của môĩ fearture ưng voi môi nhãn
        self.trungbinh = np.zeros((sl_nhan, sl_dactrung), dtype=np.float64)
        self.phuongsai = np.zeros((sl_nhan, sl_dactrung), dtype=np.float64)
        self.xacsuat_tiennghiem = np.zeros(sl_nhan, dtype=np.float64)

        for idx, giatri_nhan in enumerate(self.nhan):
            # Lâý các mâũ có nhãn băng vơi giá trị của tưng nhãn
            X_nhan = X[y == giatri_nhan]
            self.trungbinh[idx] = X_nhan.mean(axis=0)
            self.phuongsai[idx] = X_nhan.var(axis=0)
            # Xác định tân suât của nhãn đó trong tâp dư liêu
            # X_c.shape[0] sô lương mâũ của nhãn
            self.xacsuat_tiennghiem[idx] = X_nhan.shape[0] / float(sl_mau)

    def predict(self, X):
        dudoan = [self._predict(x) for x in X]
        return np.array(dudoan)

    def _predict(self, x):
        mang_xacsuat_nhan = []
        # Tính xăc suât hâụ nghiêm cho các nhãn
        for idx in range(len(self.nhan)):
            xacsuat_haunghiem = np.multiply.reduce(self.xacsuat_dieukien(idx, x)) * self.xacsuat_tiennghiem[idx]
            mang_xacsuat_nhan.append(xacsuat_haunghiem)
        # Trả vê nhãn vơi giá trị xs hâụ nghiêm max
        # print(np.multiply.reduce([1, 2, 3]))
        return self.nhan[np.argmax(mang_xacsuat_nhan)]

    def xacsuat_dieukien(self, class_idx, x):
        # print(x)
        gt_trungbinh_nhan = self.trungbinh[class_idx]
        gt_phuongsai_nhan = self.phuongsai[class_idx]
        tuso = np.exp(-((x - gt_trungbinh_nhan) ** 2) /
                      (2 * gt_phuongsai_nhan))
        mauso = np.sqrt(2 * np.pi * gt_phuongsai_nhan)
        return tuso / mauso
