import pandas as pd
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/enc_data.csv')
X = df.drop(['kaza_sonucu','kaza_ilce'],axis=1)
y = df['kaza_sonucu']
model = XGBClassifier()
model.fit(X, y)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
plot_importance(model)
pyplot.show()

X = df.drop(['kaza_sonucu','kaza_ilce'],axis=1)
y = df['kaza_sonucu']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

selection = SelectFromModel(model, prefit=True)
select_X_train = selection.transform(X_train)
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)

predictions = [round(value) for value in y_pred]
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

np.sqrt(mean_absolute_error(y_test, y_pred))