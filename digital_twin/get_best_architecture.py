import os

PATH = os.getcwd()

best_case = 'pinn_2'

os.chdir(f'{PATH}/{best_case}')
# os.system('cp pinn.py ../.')
# os.system('cp data.py ../.')

from pinn import *
from data import *
from numerical import *
import torch
import matplotlib.pyplot as plt

model = torch.load('model.pt')

files = ['exp1_6W.csv', 'exp2_6W.csv', 'exp3_6W.csv',
         'exp1_5W.csv', 'exp2_5W.csv', 'exp3_5W.csv',
         'exp1_4W.csv', 'exp2_4W.csv', 'exp3_4W.csv']

X, Y, Z, idx, idx_yf, idx_y0 = gather_data(files, 'T_train.csv')

device = torch.device('cpu')
X_train, Y_train, Z_train = put_in_device(X, Y, Z, device)

output = model(X_train)

y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0])
class parameters():
    # e = float(model.e.detach().numpy())
    # c1 = float(model.c1.detach().numpy())
    # c2 = float(model.c2.detach().numpy())
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
    T = Z_train[1806:2406].detach().numpy()
    # Q = 6
    # m_Cp = 10
prm = parameters()
t_num, y_num = euler(y0, X_train[1806:2406,0].detach().numpy(), prm)

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],0], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],0], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],0], 'o', label='Experiments 4W')
plt.plot(X_train[:241,0], output[:241,0].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(X_train[723:1084,0], output[723:1084,0].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[1806:2406,0], output[1806:2406,0].detach().numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,0], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],1], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],1], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],1], 'o', label='Experiments 4W')
plt.plot(X_train[:241,0], output[:241,1].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(X_train[723:1084,0], output[723:1084,1].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[1806:2406,0], output[1806:2406,1].detach().numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,1], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],2], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],2], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],2], 'o', label='Experiments 4W')
plt.plot(X_train[:241,0], output[:241,2].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(X_train[723:1084,0], output[723:1084,2].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[1806:2406,0], output[1806:2406,2].detach().numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,2], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('MG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[:241,0], output[:241,3].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(X_train[723:1084,0], output[723:1084,3].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[1806:2406,0], output[1806:2406,3].detach().numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,3], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('G Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[:241,0], output[:241,4].detach().numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(X_train[723:1084,0], output[723:1084,4].detach().numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(X_train[1806:2406,0], output[1806:2406,4].detach().numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,4], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('ME Concentration [mol/L]')
plt.legend()
plt.show()

# print(f'e: {float(model.e.detach().numpy())}')
# print(f'c1: {float(model.c1.detach().numpy())}')
# print(f'c2: {float(model.c2.detach().numpy())}')
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