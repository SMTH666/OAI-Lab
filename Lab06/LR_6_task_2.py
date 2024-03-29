import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# Створення моелей сигналу для навчання
i1 = np.sin(np.arange(0, 20))
i2 = np.sin(np.arange(0, 20)) * 2

t1 = np.ones([1, 20])
t2 = np.ones([1, 20]) * 2

input_data = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
target_data = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)

# Створення мережі з 2 прошарками
net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

# Ініціалізуйте початкові функції вагів
net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
net.init()

# Тренування мережі
error = net.train(input_data, target_data, epochs=500, show=100, goal=0.01)

# Запустіть мережу
output_data = net.sim(input_data)

# Побудова графіків
plt.subplot(211)
plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error (default MSE)')

plt.subplot(212)
plt.plot(target_data.reshape(80))
plt.plot(output_data.reshape(80))
plt.legend(['train target', 'net output'])
plt.show()
