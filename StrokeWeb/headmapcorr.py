import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="021000",
    database="iot"
)

data = pd.read_sql("SELECT heart_disease, avg_glucose_level, age, Residence_type, \
    smoking_status, bmi, hypertension, work_type, gender, ever_married, \
    stroke FROM stroke_final LIMIT 1000", mydb)
import math

def cov(a, b):
    result = 0
    for i in a.keys():
        result += (a[i] - np.mean(a))*(b[i] - np.mean(b))

    result /= (len(b) - 1)

    return result

def varr(a):
    result = 0
    for i in a.keys():
        result += (a[i] - np.mean(a)) ** 2

    result /= (len(a) - 1)

    return abs(result)

def Ccorr(a,b):
    result = cov(a,b) / (math.sqrt(varr(a)) * math.sqrt(varr(b)))

    return result
# print(Ccorr(data['exng'], data['cp']))
# print(data['exng'].corr(data['cp']))

# index = data.corr().nlargest(14,'exng').index
# print(index)

# dl = data.corr().nlargest(14,'cp').values[:,0]

# print(dl)

# plt.bar(index,dl)

# plt.show()

frame = []
keyword = []
for i in data.columns:
    keyword.append(i)
for i in range(len(keyword)):
    tmp = []
    for j in range(len(keyword)):
        tmp.append(Ccorr(data[keyword[i]],data[keyword[j]]))
        #tmp.append(data[keyword[i]].corr(data[keyword[j]]))
    frame.append(tmp)  
df = pd.DataFrame(frame,index=keyword,columns =keyword)
print(df)
top_corr_features = df.index
plt.figure(figsize=(14,14))
g = sns.heatmap(df,annot=True,cmap="RdYlGn")
plt.show()





# c = []
# tmpC = []
# tmp = ''

# for i in data.columns:
#     tmp += '\t' + i
#     tmpC.append(i)
    
# c.append(tmpC)
    
# print(tmp)
    
# for i in data.columns:
#     tmp = str(i)
#     tmpC = []
#     tmpC.append(i)
#     for j in data.columns:
#         value = "{:.2f}".format(Ccorr(data[i], data[j]))
#         tmp += '\t' + str(value)
#         tmpC.append(value)
#     print(tmp)
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# import pandas as pd
# diabetes = load_diabetes()
# df=pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# # print(df)
# df["Y"]=diabetes.target
# X= df.drop(["Y","sex"],axis=1)
# # "s3","age","s1","s6","s4","bp","s5","s2"
# y=df["Y"]

# X_train, X_test, Y_train, Y_test = train_test_split(X, y,train_size=0.7, random_state=0)

# model = LinearRegression()
# model.fit(X_train,Y_train)
# prediceted_y= model.predict(X_test)
# result=pd.DataFrame({'Actual':Y_test,'Predict':prediceted_y })
# print(result)
# mse= mean_squared_error(Y_test,prediceted_y)
# r2=r2_score(Y_test,prediceted_y)
# print('mse:', mse, 'r2', r2)
