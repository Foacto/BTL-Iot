import numpy as np
from collections import Counter
def entropy(y):
        mang_xuat_hien = Counter(y)
        entro = 0
        for i in mang_xuat_hien:
            p = mang_xuat_hien[i]/len(y)
            p = p* np.log2(p)
            if(p>0):
                entro = entro+ p
        return -entro

def gini(y):
    mang_xuat_hien = Counter(y)
    gini = 0
    for i in mang_xuat_hien:
        p = mang_xuat_hien[i]/len(y)
        p = p*p
        gini = gini + p
    return 1-gini    


class DecisionTree:
    def __init__(self, so_nhan_nhonhat=2, chieu_sau_toida=50, so_feats=None):
        self.so_nhan_nhonhat = so_nhan_nhonhat
        self.chieu_sau_toida = chieu_sau_toida
        self.so_feats = so_feats
        self.goc = None
        self.X_train = None
        self.y_train = None

        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if not self.so_feats:
            self.so_feats = self.X_train.shape[1]
        else:
            self.so_feats = min(self.so_feats,self.X_train.shape[1])
        self.goc = self.phat_trien(X,y)


    def phat_trien(self,X,y,do_sau=0):
        so_mau, so_features = X.shape
        so_nhan = len(np.unique(y))
        # dieu kien dung
        if(do_sau>=self.chieu_sau_toida or so_nhan == 1 or so_mau< self.so_nhan_nhonhat):
            gt_cuoicung = self.phan_tu_xuat_hien_nhieu_nhat(y)
            return self.La(gia_tri= gt_cuoicung)
        
        mang_feat = np.random.choice(so_features,self.so_feats,replace=False)

        feat_totnhat , nguong_totnhat = self.duong_di_totnhat(X,y,mang_feat)

        mang_trai, mang_phai = self.chia_mang(X[:, feat_totnhat],nguong_totnhat)
        la_trai = self.phat_trien(X[mang_trai, :],y[mang_trai],do_sau + 1)
        la_phai = self.phat_trien(X[mang_phai, :], y[mang_phai], do_sau +1)
        return self.La(feat_totnhat,nguong_totnhat, la_trai,la_phai)
    def duong_di_totnhat(self, X, y, mang_feat):
        gain_totnhat = -1
        vitri_feat_phanchia , nguong_phanchia = None,None
        for i in mang_feat:
            cot = X[:, i]
            mang_nguong = np.unique(cot)
            for nguong in mang_nguong:
                gain = self.infor_gain(y,cot,nguong)

                if(gain>gain_totnhat):
                    gain_totnhat = gain
                    vitri_feat_phanchia = i
                    nguong_phanchia = nguong
                
        return vitri_feat_phanchia, nguong_phanchia

    def gini_index(self,y,X,nguong):
        gini_cha = gini(y)
        mang_trai , mang_phai = self.chia_mang(X,nguong)
        tong_trai = len(mang_trai)
        tong_phai = len(mang_phai)
        tong = len(y)
        gini_trai = gini(mang_trai)
        gini_phai = gini(mang_phai)
        gini = gini_cha - ((tong_trai/tong)*gini_trai + (tong_phai/tong)*gini_phai)

        return gini


    def infor_gain(self, y, X, nguong):
        # tính entropy tại nút cha
        gain = 0
        entropy_cha = self.entropy(y)

        mang_trai , mang_phai = self.chia_mang(X,nguong)
        tong_trai = len(mang_trai)
        tong_phai = len(mang_phai)
        if tong_trai==0 or tong_phai==0:
            return 0
        tong = len(y)
        entropy_trai = self.entropy(y[mang_trai])
        entropy_phai = self.entropy(y[mang_phai])

        entropy_la = (tong_trai/tong) * entropy_trai + (tong_phai/tong) * entropy_phai

        gain = entropy_cha - entropy_la

        return gain

    def chia_mang(self, X, nguong):
        mang_trai = []
        mang_phai = []
        for i in range (len(X)):
            if X[i]<=nguong:
                mang_trai.append(i)
            if X[i]>=nguong:
                mang_phai.append(i)
        return np.array(mang_trai), np.array(mang_phai)

    def tim_ngon(self, x, node):
        if node.ngon_cay():
            return node.gia_tri

        if x[node.feature] <= node.nguong: # nếu giá trị nhỏ hơn ngưỡng thì đi bên trái
            return self.tim_ngon(x, node.la_trai)
        return self.tim_ngon(x, node.la_phai)
    def predict(self, X):
        mang_dudoan = []
        for x in X:
            mang_dudoan.append(self.tim_ngon(x,self.goc))
        return np.array(mang_dudoan) 
    def phan_tu_xuat_hien_nhieu_nhat(self, y):
        counter = Counter(y)
        if(len(counter)!=0):
            xuat_hien = counter.most_common(1)[0][0]
            return xuat_hien
        else:
            return [0]
    class La:
        def __init__(
        self, feature=None, nguong=None, la_trai=None, la_phai=None, *, gia_tri=None
        ):
            self.feature = feature
            self.nguong = nguong
            self.la_trai = la_trai
            self.la_phai = la_phai
            self.gia_tri = gia_tri

        def ngon_cay(self):
            return self.gia_tri is not None







    def entropy(self,y):
        hist = np.bincount(y) # liệt kê các phần tử và số lần xuất hiện của nó
        ps = hist / len(y) # tỉ lệ của các phần tử
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
