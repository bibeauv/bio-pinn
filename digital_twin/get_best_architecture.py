from pinn import *
from data import *
from numerical import *
import torch
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

best_case = 'pinn_0'

os.chdir(f'{PATH}/{best_case}')

model = torch.load('model.pt')

files = ['exp1.csv', 'exp2.csv']

X, Y, idx, idx_y0 = gather_data(files)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)

output = model(X_train)

y0 = np.array([0.596600966,0.038550212,0.000380326,0,0,34])
class parameters():
    e = float(model.e.detach().numpy())
    c1 = float(model.c1.detach().numpy())
    c2 = float(model.c2.detach().numpy())
    A1 = float(model.A1.detach().numpy())
    E1 = float(model.E1.detach().numpy())
    A2 = float(model.A2.detach().numpy())
    E2 = float(model.E2.detach().numpy())
    A3 = float(model.A3.detach().numpy())
    E3 = float(model.E3.detach().numpy())
    A4 = float(model.A4.detach().numpy())
    E4 = float(model.E4.detach().numpy())
    A5 = float(model.A5.detach().numpy())
    E5 = float(model.E5.detach().numpy())
    A6 = float(model.A6.detach().numpy())
    E6 = float(model.E6.detach().numpy())
    Q = 6
    m_Cp = 10
prm = parameters()
t_num, y_num = euler(y0, np.linspace(0,240,1000), prm)

plt.plot(X_train[idx[:4],0], Y_train[:4,0], 'o', label='Experiments 5W')
plt.plot(X_train[idx[4:],0], Y_train[4:,0], 'o', label='Experiments 6W')
plt.plot(X_train[:361,0], output[:361,0].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,0].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,0], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:4],0], Y_train[:4,1], 'o', label='Experiments 5W')
plt.plot(X_train[idx[4:],0], Y_train[4:,1], 'o', label='Experiments 6W')
plt.plot(X_train[:361,0], output[:361,1].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,1].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,1], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:4],0], Y_train[:4,2], 'o', label='Experiments 5W')
plt.plot(X_train[idx[4:],0], Y_train[4:,2], 'o', label='Experiments 6W')
plt.plot(X_train[:361,0], output[:361,2].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,2].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,2], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('MG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[:361,0], output[:361,3].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,3].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,3], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('G Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[:361,0], output[:361,4].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,4].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,4], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('ME Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:4],0], Y_train[:4,5], 'o', label='Experiments 5W')
plt.plot(X_train[idx[4:],0], Y_train[4:,5], 'o', label='Experiments 6W')
plt.plot(X_train[:361,0], output[:361,5].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[361:,0], output[361:,5].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(t_num, y_num[:,5], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel(r'Temperature [$\degree C$]')
plt.legend()
plt.show()

print(f'e: {float(model.e.detach().numpy())}')
print(f'c1: {float(model.c1.detach().numpy())}')
print(f'c2: {float(model.c2.detach().numpy())}')
print(f'A1: {float(model.A1.detach().numpy())}')
print(f'E1: {float(model.E1.detach().numpy())}')
print(f'A2: {float(model.A2.detach().numpy())}')
print(f'E2: {float(model.E2.detach().numpy())}')
print(f'A3: {float(model.A3.detach().numpy())}')
print(f'E3: {float(model.E3.detach().numpy())}')
print(f'A4: {float(model.A4.detach().numpy())}')
print(f'E4: {float(model.E4.detach().numpy())}')
print(f'A5: {float(model.A5.detach().numpy())}')
print(f'E5: {float(model.E5.detach().numpy())}')
print(f'A6: {float(model.A6.detach().numpy())}')
print(f'E6: {float(model.E6.detach().numpy())}')