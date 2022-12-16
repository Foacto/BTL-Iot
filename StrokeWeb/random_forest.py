from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np


class CustomRandomForest:
    def __init__(self, n_decisiontrees=3, soLuongLaToiThieu=2, doSauToiDa=100, soLuongFeatures=None):
        self.n_decisiontrees = n_decisiontrees
        self.soLuongLaToiThieu = soLuongLaToiThieu
        self.doSauToiDa = doSauToiDa
        self.soLuongFeatures = soLuongFeatures
        self.decisiontrees = []

    def fit(self, X, y):
        self.decisiontrees = []
        for _ in range(self.n_decisiontrees):
            cayQuyetDinh = self.CayQuyetDinh(
                soLuongLaToiThieu=self.soLuongLaToiThieu,
                doSauToiDa=self.doSauToiDa,
                soLuongFeatures=self.soLuongFeatures,
            )
            X_samp, y_samp = self.taoMauNgauNhien(X, y)
            cayQuyetDinh.fit(X_samp, y_samp)
            self.decisiontrees.append(cayQuyetDinh)

    def taoMauNgauNhien(self, X, y):
        soLuongMau = X.shape[0]
        randomIdxs = np.random.choice(soLuongMau, soLuongMau, replace=True)
        return X[randomIdxs], y[randomIdxs]

    def predict(self, X):
        tree_preds = np.array([cayQuyetDinh.predict(X)
                              for cayQuyetDinh in self.decisiontrees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [self.getValueHere(tree_pred)
                  for tree_pred in tree_preds]
        return np.array(y_pred)

    def getValueHere(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label

    class CayQuyetDinh:
        def __init__(self, soLuongLaToiThieu=2, doSauToiDa=100, soLuongFeatures=None):
            self.soLuongLaToiThieu = soLuongLaToiThieu
            self.doSauToiDa = doSauToiDa
            self.soLuongFeatures = soLuongFeatures
            self.nodeGoc = None
            self.X_train = None
            self.y_train = None

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.soLuongFeatures = X.shape[1] if not self.soLuongFeatures else min(
                self.soLuongFeatures, X.shape[1])
            self.nodeGoc = self.phatTrienCay(X, y)

        def tinhE(self, y):
            soLuongCacNhan = np.bincount(y)
            cacXacSuat = soLuongCacNhan / len(y)
            return -np.sum([xacSuat * np.log2(xacSuat) for xacSuat in cacXacSuat if xacSuat > 0])

        def predict(self, X):
            return np.array([self.duyetCay(x, self.nodeGoc) for x in X])

        def phatTrienCay(self, X, y, doSauHienTai=0):
            soLuongMau, soLuongFeatures = X.shape
            soLuongNhan = len(np.unique(y))

            if (
                doSauHienTai >= self.doSauToiDa
                or soLuongNhan == 1
                or soLuongMau < self.soLuongLaToiThieu
            ):
                leafValue = self.getValueHere(y)
                return self.Node(value=leafValue)

            featureIndexs = np.random.choice(
                soLuongFeatures, self.soLuongFeatures, replace=False)

            best_split_feat_index, best_nguong = self.duyetTheoTieuChi(
                X, y, featureIndexs)

            nhoHonNguongIdxs, lonHonNguongIdxs = self.phanTach(
                X[:, best_split_feat_index], best_nguong)

            conBenTrai = self.phatTrienCay(
                X[nhoHonNguongIdxs, :], y[nhoHonNguongIdxs], doSauHienTai + 1)
            conBenPhai = self.phatTrienCay(
                X[lonHonNguongIdxs, :], y[lonHonNguongIdxs], doSauHienTai + 1)

            return self.Node(best_split_feat_index, best_nguong, conBenTrai, conBenPhai)

        def duyetTheoTieuChi(self, X, y, featureIndexs):
            informationGainLonNhat = -1
            phanTachIdx, nguongTachTotNhat = None, None

            for feat_idx in featureIndexs:
                cotHienTai = X[:, feat_idx]
                nguongs = np.unique(cotHienTai)

                for nguong in nguongs:
                    informationGainHere = self.tinhInformationGain(
                        y, cotHienTai, nguong)

                    if informationGainHere > informationGainLonNhat:
                        informationGainLonNhat = informationGainHere
                        phanTachIdx = feat_idx
                        nguongTachTotNhat = nguong

            return phanTachIdx, nguongTachTotNhat

        def tinhInformationGain(self, y, cotHienTai, nguongTach):
            parentE = self.tinhE(y)
            nhoHonNguongIdxs, lonHonNguongIdxs = self.phanTach(
                cotHienTai, nguongTach)

            n = len(y)
            soLuongNhoHonNguong, soLuongLonHonNguong = len(
                nhoHonNguongIdxs), len(lonHonNguongIdxs)
            eLeft, eRight = self.tinhE(y[nhoHonNguongIdxs]), self.tinhE(
                y[lonHonNguongIdxs])
            weightedAvg_childE = (soLuongNhoHonNguong / n) * eLeft + \
                (soLuongLonHonNguong / n) * eRight

            informationGain = parentE - weightedAvg_childE
            return informationGain

        def phanTach(self, X_column, nguongTach):
            nhoHonNguongIdxs = np.argwhere(X_column <= nguongTach).flatten()
            lonHonNguongIdxs = np.argwhere(X_column >= nguongTach).flatten()
            return nhoHonNguongIdxs, lonHonNguongIdxs

        def duyetCay(self, x, node):
            if node.laNodeLa():
                return node.value

            if x[node.feature] <= node.nguong:
                return self.duyetCay(x, node.conBenTrai)
            return self.duyetCay(x, node.conBenPhai)

        def getValueHere(self, y):
            counter = Counter(y)
            label = counter.most_common(1)[0][0]
            return label

        class Node:
            def __init__(
                self, feature=None, nguong=None, conBenTrai=None, conBenPhai=None, *, value=None
            ):
                self.feature = feature
                self.nguong = nguong
                self.conBenTrai = conBenTrai
                self.conBenPhai = conBenPhai
                self.value = value

            def laNodeLa(self):
                return self.value is not None


if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv("./stroke_final.csv")

    features = ["gender", "age", "hypertension", "smoking_status"]

    X = data[features].to_numpy()[0: 1000]
    y = data["stroke"].to_numpy()[0: 1000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = CustomRandomForest(n_decisiontrees=3, doSauToiDa=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)

# print('nguongTach', nguongTach)
    # if nguongTach == None:
    #     nguongTach = 0
