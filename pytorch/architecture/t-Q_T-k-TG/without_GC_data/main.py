from data import *
from pinn import *
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.getcwd() + '/model.pt'

# Data parameters
device = torch.device('cpu')
y0 = torch.from_numpy(np.array([0.6, 0.6, 0.02, 0.02]).reshape(-1,1)).float().to(device)

# Parameters
class parameters():
    dHrx = -45650              # J/mol
    epsilon = 0.1              # -
    V = 6.3                    # mL
    m = 5.6                    # g
    Cp = 2                     # J/g/C
prm = parameters()

# Inputs
files = ['lowT-mass-4W_2302071732061_MwData.csv', 'lowT-mass-5W_2302071731344_MwData.csv']
t_data, T_ir, P = read_data(files)

t_train = np.array(t_data)
T_train = np.array(T_ir)
Q_train = np.array(P)

start = [0]
for i in range(len(t_train)-1):
    if (t_train[i+1] - t_train[i]) < 0:
        start.append(i+1)
idx1_ = np.arange(start[0],start[1])
idx1 = idx1_[::60]
idx2_ = np.arange(start[1],len(t_train))
idx2 = idx2_[::60]
idx = idx1.tolist() + idx2.tolist()
idx_y0 = start + [start[1]-1] + [len(t_train)-1]

X = np.append(t_train.reshape(-1,1), Q_train.reshape(-1,1), axis=1)
Y = T_train.reshape(-1,1)

# Train data
X_train, Y_train = put_in_device(X, Y, device)

# Create PINN
f_hat = torch.zeros(X_train.shape[0],1).to(device)
PINN = Curiosity(X_train, Y_train, y0, idx, idx_y0, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

# Training
epochs = 50000
epoch = 1
vec_loss = []
while epoch <= epochs:
    PINN.optimizer.step(PINN.closure)
    vec_loss.append(float(PINN.loss(X_train, Y_train).detach().numpy()))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t total_loss: {PINN.total_loss:.2e} \t T_data_loss: {PINN.loss_T_data:.2e} \t cTG_data_loss: {PINN.loss_cTG_data:.2e} \t T_ode_loss: {PINN.loss_T_ode:.2e} \t cTG_ode_loss: {PINN.loss_cTG_ode:.2e}')
    if epoch == 10000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)
    if epoch == 25000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)
    epoch += 1

PINN.save_model(PATH)

plt.plot(vec_loss)
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), PINN.PINN(X_train)[idx1_,1].detach().numpy())
plt.plot(X_train[idx1,0].detach().numpy(), T_train[idx1], 'o')
plt.plot(X_train[idx2_,0].detach().numpy(), PINN.PINN(X_train)[idx2_,1].detach().numpy())
plt.plot(X_train[idx2,0].detach().numpy(), T_train[idx2], 'o')
plt.xlabel('Time [min]')
plt.ylabel(r'Temperature [$\degree$C]')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), PINN.PINN(X_train)[idx1_,0].detach().numpy())
plt.plot(X_train[idx2_,0].detach().numpy(), PINN.PINN(X_train)[idx2_,0].detach().numpy())
plt.xlabel('Time [min]')
plt.ylabel(r'Concentration [mol/L]')
plt.show()

plt.plot(T_train[idx1], PINN.PINN(X_train)[idx1,2].detach().numpy())
plt.plot(T_train[idx2], PINN.PINN(X_train)[idx2,2].detach().numpy())
plt.xlabel(r'Temperature [$\degree$C]')
plt.ylabel(r'Kinetic constant [s$^{-1}$]')
plt.show()