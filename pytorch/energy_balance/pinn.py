from preprocess import *
from source import *
from postprocess import *
import matplotlib.pyplot as plt

# Create device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read and prepare data
collocation_points = 100 * 10 + 1
t1 = 0.
t2 = 10.
T1 = 25.
T2 = 70.
Ea = 200.
A = 50.
prm = [A*np.exp(-Ea/T1), 0.3, 1, -50]
t_train, \
T_train, \
cTG_train, \
y_pred, idx = read_data(Ea, A, t1, t2, T1, T2, collocation_points, device, prm)

# ODEs residual
f_hat = torch.zeros(t_train.shape[0],1).to(device)

# Sub PINNs
Ea_pred = 100.
A_pred = 10.
k_pred = [Ea_pred, A_pred]
class parametres():
    Q = 0.3
    V = 1
    dHrx = -50
prm_mw = parametres()
PINN_cTG = cTGNN(device, f_hat, y_pred, k_pred, t_train, cTG_train, idx)
PINN_Temp = TempNN(device, f_hat, y_pred, k_pred, t_train, T_train, prm_mw, idx)

x_train = t_train

# Initial predictions
y_pred['cTG'] = PINN_cTG.dnn(x_train).detach().clone().flatten()
y_pred['Temp'] = PINN_Temp.dnn(x_train).detach().clone().flatten()

# Training
epoch = 0
max_epochs = 150000
vec_loss_cTG = []
vec_loss_Temp = []
while epoch < max_epochs:
    # Backward
    train_cNN(PINN_cTG, 'cTG', y_pred, k_pred, t_train)
    vec_loss_cTG.append(float(PINN_cTG.loss(x_train, cTG_train).detach().numpy()))
    
    train_cNN(PINN_Temp, 'Temp', y_pred, k_pred, t_train)
    vec_loss_Temp.append(float(PINN_Temp.loss(x_train, T_train).detach().numpy()))

    epoch += 1
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t cTG_loss: {vec_loss_cTG[-1]:.4e} \t Temp_loss: {vec_loss_Temp[-1]:.4e}')
    
    if epoch == 10000:
        PINN_cTG.optimizer = torch.optim.Adam(PINN_cTG.params, lr=1e-3)
        PINN_Temp.optimizer = torch.optim.Adam(PINN_Temp.params, lr=1e-3)

    if epoch == 80000:
        PINN_cTG.optimizer = torch.optim.Adam(PINN_cTG.params, lr=1e-4)
        PINN_Temp.optimizer = torch.optim.Adam(PINN_Temp.params, lr=1e-4)

    if epoch == 120000:
        PINN_cTG.optimizer = torch.optim.Adam(PINN_cTG.params, lr=1e-5)
        PINN_Temp.optimizer = torch.optim.Adam(PINN_Temp.params, lr=1e-5)

print('\n')

plt.plot(vec_loss_cTG, label='loss_cTG')
plt.plot(vec_loss_Temp, label='loss_Temp')
plt.xscale('log')
plt.yscale('log')
plt.show()

dt = t2 / (collocation_points - 1)
Ea_euler = PINN_cTG.dnn.Ea.detach().numpy()[0]
A_euler = PINN_cTG.dnn.A.detach().numpy()[0]
t_num, y_num = euler_explicite([0.6, 25.], Ea_euler, A_euler, dt, t2, prm)
t_num2, y_num2 = euler_explicite([0.6, 25.], Ea, A, dt, t2, prm)

fig, axs = plt.subplots(2)
axs[0].plot(t_train.detach().numpy(), PINN_cTG.dnn(x_train).detach().numpy(), label='PINN')
axs[0].plot(t_num, y_num[:,0], '--', label='Euler')
axs[0].plot(t_num2, y_num2[:,0], label='Real')
axs[0].set_ylabel('Concentration [mol/L]')
axs[0].legend()
axs[1].plot(t_train.detach().numpy(), PINN_Temp.dnn(x_train).detach().numpy(), label='PINN')
axs[1].plot(t_num, y_num[:,1], '--', label='Euler')
axs[1].plot(t_num2, y_num2[:,1], label='Real')
axs[1].set_ylabel('Température [degC]')
axs[1].set_xlabel('Temps [min]')
axs[1].legend()
plt.show()

print(PINN_cTG.dnn.Ea)
print(PINN_cTG.dnn.A)
