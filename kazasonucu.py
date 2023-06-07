import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.reindex(columns=['sira_no', 'kaza_il', 'kaza_ilce', 'koy_mahalle', 'yol_sinifi',
       'yol_tipi', 'yol_kpln', 'yerlesim_yeri', 'tarih', 'gun', 'saat',
       'serit_banket', 'serit_durum', 'levha_durun', 'isik_durum',
       'aydinlatma', 'gun_durum', 'hava_durumu', 'calisma_durumu',
       'yol_cls_isaret', 'yol_yuzeyi', 'yatay_gzr', 'dusey_gzr',
       'kavsak_durum', 'gecit_durum', 'diger', 'olus_1', 'olus_2',
       'ilk_carp_yeri' , 'ihlal1', 'ihlal2', 'arac_cinsi',
       'hasar_der', 'yanma', 'yakit', 'coord','kaza_sonucu'])
test = test.reindex(columns=['sira_no', 'kaza_il', 'kaza_ilce', 'koy_mahalle', 'yol_sinifi',
       'yol_tipi', 'yol_kpln', 'yerlesim_yeri', 'tarih', 'gun', 'saat',
       'serit_banket', 'serit_durum', 'levha_durun', 'isik_durum',
       'aydinlatma', 'gun_durum', 'hava_durumu', 'calisma_durumu',
       'yol_cls_isaret', 'yol_yuzeyi', 'yatay_gzr', 'dusey_gzr',
       'kavsak_durum', 'gecit_durum', 'diger', 'olus_1', 'olus_2',
       'ilk_carp_yeri', 'ihlal1', 'ihlal2', 'arac_cinsi',
       'hasar_der', 'yanma', 'yakit', 'coord','kaza_sonucu'])

train = train.drop(['sira_no', 'kaza_il', 'tarih', 'saat', 'coord'],axis=1)
test = test.drop(['sira_no', 'kaza_il', 'tarih', 'saat', 'coord'],axis=1)

from sklearn.preprocessing import LabelEncoder

traindf = train.copy()
traindf = traindf.astype(str)
testdf = test.copy()
testdf = testdf.astype(str)

le = LabelEncoder()
traindf = traindf.apply(lambda col: le.fit_transform(col))
traindf.head(10)

testdf = testdf.apply(lambda col: le.fit_transform(col))
testdf.head(10)

X = traindf.drop('kaza_sonucu', axis=1)
y = traindf['kaza_sonucu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cbr = CatBoostRegressor()
cbr.fit(X_train, y_train)

cbr_prediction = cbr.predict(X_test)

plt.figure(figsize=(20, 8))
plt.plot([i for i in range(len(y_test))],y_test, label="actual values")
plt.plot([i for i in range(len(y_test))],cbr_prediction, label="Predicted values")
plt.legend()
plt.show()

plt.figure(figsize=(20, 8))
print("Mean of actual values : ", y_test.mean())
print("Mean of predicted values: ", cbr_prediction.mean())
plt.plot([i for i in range(len(y_test))],y_test, label="actual values")
plt.plot([i for i in range(len(y_test))],cbr_prediction, label="Predicted values")
plt.plot([i for i in range(len(y_test))],[y_test.mean() for x in range(len(y_test))], label = "mean of actual values")
plt.plot([i for i in range(len(y_test))],[cbr_prediction.mean() for y in range(len(y_test))], label = 'mean of predicted values')
plt.legend()
plt.show()

from sklearn.metrics import  r2_score
print('R-square score is :', r2_score(y_test, cbr_prediction))

from catboost import CatBoostClassifier
cbc = CatBoostClassifier()
cbc.fit(X_train, y_train)

cbc_pred = cbc.predict(X_test)

sns.set(rc={'figure.figsize':(11.7,8.27)})
cm = confusion_matrix(y_test, cbc_pred)
sns.heatmap(cm,annot=True)
plt.savefig('confusion_Matrix.png')

from sklearn.metrics import accuracy_score
accuracy_score(y_test, cbc_pred)

print(le.inverse_transform(y_test[:10]))
print(le.inverse_transform(cbc.predict(X_test[:10])))