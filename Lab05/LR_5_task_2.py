import numpy as np

def sigmoid(x):
    # Функція активації: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Вхідні дані про вагу, додавання зміщення і застосування функції активації
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class YasenNeuralNetwork:
    def __init__(self):
        weights_hidden = np.array([0, 1])
        bias_hidden = 0
        weights_output = np.array([0, 1])
        bias_output = 0

        # Ініціалізація нейронів у прихованому та вихідному шарах
        self.hidden1 = Neuron(weights_hidden, bias_hidden)
        self.hidden2 = Neuron(weights_hidden, bias_hidden)
        self.output = Neuron(weights_output, bias_output)

    def feedforward(self, x):
        # Виклик функції feedforward для кожного нейрона та передача вихідних значень між ними
        out_hidden1 = self.hidden1.feedforward(x)
        out_hidden2 = self.hidden2.feedforward(x)
        out_output = self.output.feedforward(np.array([out_hidden1, out_hidden2]))
        return out_output

def main():
    # Приклад використання нейронної мережі
    x = np.array([2, 3])

    # Створення та використання об'єкта нейронної мережі
    network = YasenNeuralNetwork()
    output = network.feedforward(x)
    print("Network Output:", output)

if __name__ == "__main__":
    main()
