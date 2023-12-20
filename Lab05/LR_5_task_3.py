import numpy as np
import matplotlib.pyplot as plt

# Завантаження вхідних даних
text = np.loadtxt('data_perceptron.txt')

# Поділ точок даних та міток
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Размерность 1')
plt.ylabel('Размерность 2')
plt.title('Входные данные')

# Визначення функції активації (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Ініціалізація ваг та зміщення
weights = np.zeros((2, 1))
bias = 0

# Визначення функції для визначення виходу перцептрону
def perceptron_output(inputs):
    return sigmoid(np.dot(inputs, weights) + bias)

# Визначення функції втрат (середньоквадратична помилка)
def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Визначення градієнта функції втрат по вагам
def compute_gradient(inputs, predictions, targets):
    error = predictions - targets
    gradient_weights = np.dot(inputs.T, error)
    gradient_bias = np.sum(error)
    return gradient_weights, gradient_bias

# Тренування перцептрону
learning_rate = 0.03
num_epochs = 100
error_progress = []

for epoch in range(num_epochs):
    # Обчислення виходу перцептрону
    predictions = perceptron_output(data)

    # Обчислення та виведення значення функції втрат
    error = mean_squared_error(predictions, labels)
    error_progress.append(error)

    # Обчислення та виведення градієнту
    gradient_weights, gradient_bias = compute_gradient(data, predictions, labels)

    # Оновлення ваг та зміщення згідно градієнту
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
