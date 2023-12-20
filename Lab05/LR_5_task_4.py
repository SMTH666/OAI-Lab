import numpy as np
import matplotlib.pyplot as plt

# Завантаження вхідних даних
text = np.loadtxt('data_simple_nn.txt')

# Поділ точок даних та міток
data = text[:, 0:2]
labels = text[:, 2:]

# Побудова графіка вхідних даних
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Размерность 1')
plt.ylabel('Размерность 2')
plt.title('Входные данные')

# Мінімальне та максимальне значення для кожного виміру
dim1_min, dim1_max = data[:, 0].min(), data[:, 0].max()
dim2_min, dim2_max = data[:, 1].min(), data[:, 1].max()

# Визначення кількості нейронів у вихідному шарі
num_output = labels.shape[1]

# Визначення ваг та зміщення нейронів
weights = np.random.rand(2, num_output)
bias = np.zeros((1, num_output))

# Визначення функції активації (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Визначення функції для визначення виходу мережі
def predict(inputs, weights, bias):
    total = np.dot(inputs, weights) + bias
    return sigmoid(total)

# Визначення функції втрат (середньоквадратична помилка)
def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Тренування мережі
learning_rate = 0.03
num_epochs = 100
error_progress = []

for epoch in range(num_epochs):
    # Обчислення виходу мережі
    predictions = predict(data, weights, bias)

    # Обчислення та виведення значення функції втрат
    error = mean_squared_error(predictions, labels)
    error_progress.append(error)

    # Обчислення та виведення градієнту
    gradient_weights = np.dot(data.T, (predictions - labels) * predictions * (1 - predictions))
    gradient_bias = np.sum((predictions - labels) * predictions * (1 - predictions), axis=0)

    # Оновлення ваг та зміщення
    weights -= learning_rate * gradient_weights
    bias -= learning_rate * gradient_bias

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {error:.4f}')

# Побудова графіка процесу навчання
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')
plt.grid()
plt.show()

# Виконання класифікатора на тестових точках даних
print('\nTest results:')
data_test = np.array([[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]])
for item in data_test:
    print(item, '-->', predict(item, weights, bias)[0])
