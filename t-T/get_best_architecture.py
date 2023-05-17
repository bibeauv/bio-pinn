from pinn import *
from data import *
from numerical import *
import torch
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

os.chdir(f'{PATH}/pinn_1')

model = torch.load('model.pt')

files = ['T_train.csv']

X, Y = gather_data(files)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)

output = model(X_train)

y0 = np.array([34.2])
class parameters():
    e = float(model.e.detach().numpy())
    c1 = float(model.c1.detach().numpy())
    c2 = float(model.c2.detach().numpy())
    Q = 4
    m_Cp = 10
prm = parameters()
t_num, y_num = euler(y0, np.linspace(0,600,1000), prm)

plt.plot(X_train[:,0], Y_train[:,0], 'o', markersize=1, label='Experiments')
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
T_df.to_csv('new_T_train.csv')