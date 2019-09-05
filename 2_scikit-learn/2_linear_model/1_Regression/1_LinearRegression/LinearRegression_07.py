import pandas as pd

path = '../../../../data/score.csv'

score_r = pd.read_csv(path)

print(score_r.info())
print(score_r.describe())

X = score_r.iloc[:,2:]
y = score_r.iloc[:,1:2]
X_train = X.values
y_train = y.values


# 학습
from sklearn.neighbors import KNeighborsRegressor
knr_model = KNeighborsRegressor(n_neighbors=2,n_jobs=-1).fit(X_train,y_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression(n_jobs=-1).fit(X_train,y_train)

# 평가
knr_predicted = knr_model.predict(X_train)
predicted = model.predict(X_train)

# 평균제곱오차
from sklearn.metrics import mean_squared_error
knr_mse = mean_squared_error(y,knr_predicted)
mse = mean_squared_error(y,predicted)

# 평균절대오차 
from sklearn.metrics import mean_absolute_error
knr_mae = mean_absolute_error(y,knr_predicted)
mae = mean_absolute_error(y,predicted)

print('\n')
print('KNR 평가 (R2) : ', knr_model.score(X_train,y_train))
print('LR  평가 (R2) : ', model.score(X_train,y_train))
print('\n')
print('KNR 평균절대오차 : ',knr_mae)
print('LR  평균절대오차 : ',mae)
print('\n')
print('KNR 평균제곱오차 : ',knr_mse)
print('LR  평균제곱오차 : ',mse)
