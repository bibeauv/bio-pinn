from pinn import *
from data import *
from numerical import *
import torch
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

loss_dict = {'case': [], 'loss_c_data': [], 'loss_c_ode': [], 'loss_T_data': [], 'loss_T_ode': []}

for dirpath, dirnames, filenames in os.walk(PATH):

    if 'model.pt' in filenames and 'loss.txt' in filenames:

        os.chdir(dirpath)

        with open('loss.txt', 'r') as f:
            lines = f.readlines()
            line = lines[0]
            loss_info = line.split('\t')
            loss_c_data = float(loss_info[1].split(':')[1])
            loss_c_ode = float(loss_info[3].split(':')[1])
            loss_T_data = float(loss_info[2].split(':')[1])
            loss_T_ode = float(loss_info[4].split(':')[1])
            loss_dict['loss_c_data'].append(loss_c_data)
            loss_dict['loss_c_ode'].append(loss_c_ode)
            loss_dict['loss_T_data'].append(loss_T_data)
            loss_dict['loss_T_ode'].append(loss_T_ode)
            loss_dict['case'].append(dirpath.split('/')[-1])

loss_df = pd.DataFrame(loss_dict)
best_loss_df = loss_df[loss_df['loss_c_ode'] == loss_df['loss_c_ode'].min()]
best_case = best_loss_df['case'].to_string(index=False)

print(loss_df)

os.chdir(f'{PATH}/{best_case}')

model = torch.load('model.pt')

files = ['exp1.csv']

collocation_points = 721
X, Y, idx, idx_y0 = gather_data(files, collocation_points)

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
t_num, y_num = euler(y0, X_train[:721].detach().numpy().flatten(), prm)

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

plt.plot(X_train[idx], Y_train[:,5], 'o', label='Experiments')
plt.plot(X_train, output[:,5].detach().numpy(), 'o', markersize=1, label='PINN')
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