import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Масштабування ознак для поліпшення роботи лінійної регресії
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("Linear regressor performance:")
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("R2 score:", round(r2_score(y_test, y_pred), 2))
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean squared error:", round(mean_squared_error(y_test, y_pred), 2))

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
