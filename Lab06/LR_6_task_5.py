import numpy as np
import neurolab as nl

target = [[0, 1, 1, 1, 0,
           0, 1, 0, 1, 0,
           0, 1, 1, 1, 0,  # P
           0, 1, 0, 0, 0,
           0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0,
           0, 1, 0, 1, 0,
           1, 1, 1, 1, 1,  # A
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0,
           0, 1, 1, 1, 0,
           0, 0, 1, 0, 0,  # I
           0, 0, 1, 0, 0,
           0, 1, 1, 1, 0]
          ]
chars = ['Y', 'A', 'Y']
target = np.asfarray(target)
target[target == 0] = -1
net = nl.net.newhop(target)
output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())
print("\nTest on defaced А:")
test = np.asfarray(
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 1, 1, 1, 1,  # A
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1],
)
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))
