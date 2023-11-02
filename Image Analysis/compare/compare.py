import numpy as np

rui = np.load('data/83047_T4.npy')
ze = np.load('data/maiav2.npy')

print(rui.shape)
print(ze.shape)


if rui.shape != ze.shape:
    print('SHAPES DIFERENTES')
counter = 0
for i in range(rui.shape[0]):
    if (rui[i] != ze[i]):
        counter += 1

print(f'Diferencas: {counter * 100 / rui.shape[0]} %')
        
