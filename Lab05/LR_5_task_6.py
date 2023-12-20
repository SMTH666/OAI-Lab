import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Генерація тренувальних даних
min_val = -15
max_val = 15
num_points = 130
x = np. linspace(min_val, max_val, num_points)
y = 2 * x**2 + 2 * x + 1
y /= np.linalg.norm(y)

# Створення даних та міток
data = x.reshape(num_points,1)
labels = y.reshape(num_points,1)

#Побудуємо графік вхідних даних.
plt. figure()
plt.scatter(data,labels)
plt. xlabel('Размерность 1 ')
plt.ylabel('Размерность 2')
plt. title('Входные данные')

# Вихідний шар складається з одного нейрона.
nn = nl.net.newff([[min_val, max_val]],[3,3,1])

# Завдання градієнтного спуску як навчального алгоритму
nn.trainf = nl.train.train_gd

# Тренування нейронної мереж
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Виконання нейронної мережі на тренувальних даних
output = nn.sim(data)
y_pred = output.reshape(num_points)

# Побудова графіка помилки навчання
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')

# Побудова графіка результатів
x_dense = np. linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt. figure ()
plt.plot(x_dense, y_dense_pred, '-',x, y,'.',x,y_pred,'p')
plt.title('фактические и прогнозные значения')
plt.show()
