from pinn import *
from data import *
from numerical import *
import torch
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

loss_dict = {'case': [], 'loss_T_data': [], 'loss_T_ode': []}

for dirpath, dirnames, filenames in os.walk(PATH):

    if 'model.pt' in filenames and 'loss.txt' in filenames:

        os.chdir(dirpath)

        with open('loss.txt', 'r') as f:
            lines = f.readlines()
            line = lines[0]
            loss_info = line.split('\t')
            loss_T_data = float(loss_info[1].split(':')[1])
            loss_T_ode = float(loss_info[2].split(':')[1])
            loss_dict['loss_T_data'].append(loss_T_data)
            loss_dict['loss_T_ode'].append(loss_T_ode)
            loss_dict['case'].append(dirpath.split('/')[-1])

loss_df = pd.DataFrame(loss_dict)
best_loss_df = loss_df[loss_df['loss_T_ode'] == loss_df['loss_T_ode'].min()]
best_case = best_loss_df['case'].to_string(index=False)

print(loss_df)

os.chdir(f'{PATH}/{best_case}')

model = torch.load('model.pt')

files = ['exp1.csv', 'exp2.csv', 'exp3.csv']

X, Y, idx, idx_y0 = gather_data(files)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)

output = model(X_train)

y0 = np.array([34])
class parameters():
    e = float(model.e.detach().numpy())
    c1 = float(model.c1.detach().numpy())
    c2 = float(model.c2.detach().numpy())
    Q = 6
    m_Cp = 10
prm = parameters()
t_num, y_num = euler(y0, np.linspace(0,600,1000), prm)

plt.plot(X_train[idx,0], Y_train[:,0], 'o', label='Experiments')
plt.plot(X_train[:,0], output[:,0].detach().numpy(), 'o', markersize=1, label='PINN')
plt.plot(t_num, y_num[:,0], '--', label='Numerical')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

print(f'e: {float(model.e.detach().numpy())}')
print(f'c1: {float(model.c1.detach().numpy())}')
print(f'c2: {float(model.c2.detach().numpy())}')

Temperature = output[:,0].detach().numpy().flatten()
Time = X_train[:,0].detach().numpy().flatten()
Power = X_train[:,1].detach().numpy().flatten()
T_df = {'Time': Time, 'Power': Power, 'Temperature': Temperature}
T_df = pd.DataFrame.from_dict(T_df)
T_df.to_csv('T_train.csv')