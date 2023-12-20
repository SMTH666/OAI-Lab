import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

fig, ax = plt.subplots()
ax.scatter(X, y, edgecolors=(0, 0, 0))
plt.show()

# Трансформація ознак за допомогою поліноміальних ознак
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("X[0] =", X[0])
print("X[1] =", X[1])
print("Y[1] =", y[1])

# Навчання лінійної регресії на поліноміальних ознаках
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Перетин:", lin_reg.intercept_)
print("Коефіцієнти регресії:", lin_reg.coef_)

# Передбачення на основі навченої моделі
y_pred = lin_reg.predict(X_poly)

fig, ax = plt.subplots()
ax.scatter(X, y, edgecolors=(0, 0, 0))
plt.plot(X, y_pred, color='red', linewidth=4)
plt.show()
