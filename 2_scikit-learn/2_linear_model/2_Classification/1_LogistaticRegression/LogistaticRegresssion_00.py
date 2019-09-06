import numpy as np
from matplotlib import pyplot as plt

X = np.arange(1,11).reshape(-1,1)
y = np.array([1 if data > 5 else 0 for data in X])
y[3] = 1
print(X)
print(y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs').fit(X,y)

print('학습 평가 : ',model.score(X,y))

plt.figure(figsize=(7,7))
for index , data in enumerate(X) :
  plt.scatter(data,data,c='k',
              marker='x' if y[index] == 0 else 'D'
              )

plt.grid(True)
plt.show()

print('예측 결과 : ',model.predict(X))
print('예측 결과 : ', X * model.coef_ + model.intercept_)