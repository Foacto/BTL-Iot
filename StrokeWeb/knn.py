from collections import Counter

import numpy as np

class Custom_KNN:
    def __init__(self, k):
        self.counterX = 0
        self.k = k
        self.xac_suat_cuoi = 0

    def _khoangcach_euclideane(self, x1, x2):
        # Tìm khoảng cách giữa 2 mẫu
        sum = 0
        
        for i in range(len(x1)):
            sum += (x1[i] - x2[i]) ** 2

        return np.sqrt(sum)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        list_kq = [self._predict(x) for x in X]
        return np.array(list_kq)

    def _predict(self, x):
        # if (self.counterX % 1000) == 0:
        #     print(self.counterX)
        # self.counterX += 1

        # Tính khoảng cách giữa x và tất cả các mẫu trong bộ dữ liệu
        khoangcach = [self._khoangcach_euclideane(x, x_train) for x_train in self.X_train]

        # sắp xếp theo chiều tăng dần giá trị khoảng cách và trả về mảng chứa vị trí của các mẫu sau khi sắp xếp
        k_mau_gan_nhat = np.argsort(khoangcach)
        # Lấy vị trí k mẫu gần với giá trị nhất
        k_mau_gan_nhat = k_mau_gan_nhat[: self.k]

        # Lấy tập các nhãn của k mẫu đó
        k_nhan_gan_nhat = [self.y_train[i] for i in k_mau_gan_nhat]
        # print(k_nhan_gan_nhat)

        # trả về nhãn có số lần xuất hiện nhiều nhất
        ket_qua = Counter(k_nhan_gan_nhat).most_common(1)

        # Xác xuất dự đoán đúng
        self.xac_suat_cuoi = ket_qua[0][1] / self.k
        # print(self.xac_suat_cuoi)

        return ket_qua[0][0]
