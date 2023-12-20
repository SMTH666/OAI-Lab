import numpy as np
import neurolab as nl

target = np.array([[-1, 1, -1, -1, 1, -1, -1, 1, -1],
                   [1, 1, 1, 1, -1, 1, 1, -1, 1],
                   [1, -1, 1, 1, 1, 1, 1, -1, 1],
                   [1, 1, 1, 1, -1, -1, 1, -1, -1],
                   [-1, -1, -1, -1, 1, -1, -1, -1, -1]])

input_data = np.array([[-1, -1, 1, 1, 1, 1, 1, -1, 1],
                       [-1, -1, 1, -1, 1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, 1, -1, -1, 1, -1]])

# Створення та тренування нейромережі
net = nl.net.newhem(target)

# Тестування на тренувальних зразках
output_train = net.sim(target)
print("Test on train samples (must be [0, 1, 2, 3, 4]):")
print(np.argmax(output_train, axis=0))

# Тестування на першому тестовому зразку
output_recurrent = net.sim([input_data[0]])
print("Outputs on recurrent cycle:")
print(np.array(net.layers[1].outs))

# Тестування на всіх тестових зразках
output_test = net.sim(input_data)
print("Outputs on test samples:")
print(output_test)
