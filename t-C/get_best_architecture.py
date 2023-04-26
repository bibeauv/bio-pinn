from pinn import *
from data import *
from numerical import *
import torch
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

loss_dict = {'case': [], 'loss_c_data': [], 'loss_c_ode': []}

for dirpath, dirnames, filenames in os.walk(PATH):

    if 'model.pt' in filenames and 'loss.txt' in filenames:

        os.chdir(dirpath)

        with open('loss.txt', 'r') as f:
            lines = f.readlines()
            line = lines[0]
            loss_info = line.split('\t')
            loss_c_data = float(loss_info[1].split(':')[1])
            loss_c_ode = float(loss_info[2].split(':')[1])
            loss_dict['loss_c_data'].append(loss_c_data)
            loss_dict['loss_c_ode'].append(loss_c_ode)
            loss_dict['case'].append(dirpath.split('/')[-1])

loss_df = pd.DataFrame(loss_dict)
best_loss_df = loss_df[loss_df['loss_c_ode'] == loss_df['loss_c_ode'].min()]
best_case = best_loss_df['case'].to_string(index=False)

print(loss_df)

os.chdir(f'{PATH}/pinn_0')

model = torch.load('model.pt')

files = ['exp1.csv', 'exp2.csv', 'exp3.csv']

X, Y, Z, idx, idx_y0 = gather_data(files, 'T_train.csv', 6.0)

device = torch.device('cpu')
X_train, Y_train, Z_train = put_in_device(X, Y, Z, device)

output = model(X_train)

y0 = np.array([0.596600966,0.038550212,0.000380326,0,0])
class parameters():
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
    T = Z_train[:241].detach().numpy().flatten()
prm = parameters()
t_num, y_num = euler(y0, X_train[:241].detach().numpy().flatten(), prm)

plt.plot(X_train[idx], Y_train[:,0], 'o', label='Experiments')
plt.plot(X_train, output[:,0].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,0], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx], Y_train[:,1], 'o', label='Experiments')
plt.plot(X_train, output[:,1].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,1], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx], Y_train[:,2], 'o', label='Experiments')
plt.plot(X_train, output[:,2].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,2], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('MG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train, output[:,3].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,3], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('G Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train, output[:,4].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,4], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('ME Concentration [mol/L]')
plt.legend()
plt.show()

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
