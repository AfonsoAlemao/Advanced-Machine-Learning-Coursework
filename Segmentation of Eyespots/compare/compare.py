import numpy as np

rui = np.load('data/rui.npy')
outro = np.load('data/ytest_Classification2.npy')

print(rui.shape)
print(outro.shape)


if rui.shape != outro.shape:
    print('SHAPES DIFERENTES')
counter = 0
for i in range(rui.shape[0]):
    if (rui[i] != outro[i]):
        counter += 1

print(f'Diferencas: {counter * 100 / rui.shape[0]} %')
        
