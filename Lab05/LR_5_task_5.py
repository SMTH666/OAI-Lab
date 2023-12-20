import numpy as np
import matplotlib.pyplot as plt

# Генерація тренувальних даних
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Створення даних та міток
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# Побудова графіка вхідних даних.
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Размерность 1')
plt.ylabel('Размерность 2')
plt.title('Входные данные')

# Визначення моделі нейронної мережі
class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(1, 10)
        self.weights2 = np.random.rand(10, 6)
        self.weights3 = np.random.rand(6, 1)
        self.bias1 = np.zeros((1, 10))
        self.bias2 = np.zeros((1, 6))
        self.bias3 = np.zeros((1, 1))

    def forward(self, x):
        self.layer1 = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        output = np.dot(self.layer2, self.weights3) + self.bias3
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, x, y, output, learning_rate):
        error = y - output
        output_delta = error * self.sigmoid_derivative(output)
        error_layer2 = output_delta.dot(self.weights3.T)
        layer2_delta = error_layer2 * self.sigmoid_derivative(self.layer2)
        error_layer1 = layer2_delta.dot(self.weights2.T)
        layer1_delta = error_layer1 * self.sigmoid_derivative(self.layer1)

        self.weights3 += learning_rate * self.layer2.T.dot(output_delta)
        self.bias3 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights2 += learning_rate * self.layer1.T.dot(layer2_delta)
        self.bias2 += learning_rate * np.sum(layer2_delta, axis=0, keepdims=True)
        self.weights1 += learning_rate * x.T.dot(layer1_delta)
        self.bias1 += learning_rate * np.sum(layer1_delta, axis=0, keepdims=True)

# Ініціалізація моделі, втрат та оптимізатора
neural_net = NeuralNetwork()

# Тренування нейронної мережі
num_epochs = 2000
error_progress = []
learning_rate = 0.01

for epoch in range(num_epochs):
    # Обчислення виходу моделі та оновлення ваг
    output = neural_net.forward(data)
    neural_net.backward(data, labels, output, learning_rate)

    # Обчислення та виведення значення функції втрат
    loss = np.mean(np.square(labels - output))
    error_progress.append(loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

# Виконання нейронної мережі на тренувальних даних
neural_net.forward(data)
y_pred = neural_net.layer2

# Побудова графіків
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')

plt.figure()
plt.plot(x, y, '.', label='Фактичні значення')
plt.plot(x, y_pred, 'p', label='Прогнозовані значення')
plt.legend()
plt.title('Фактичні та прогнозовані значення')
plt.show()
