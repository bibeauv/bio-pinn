from data import *
from pinn import *
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.getcwd() + '/model.pt'

# Data parameters
device = torch.device('cpu')
y0 = torch.from_numpy(np.array([0.6, 0.6,
                                0.03, 0.03]).reshape(-1,1)).float().to(device)

# Parameters
class parameters():
    dHrx = -45650              # J/mol
    epsilon = 0.1              # -
    V = 6.3                    # mL
    m = 5.6                    # g
    Cp = 2                     # J/g/C
prm = parameters()

# Inputs (moyenne mobile)
files = ['lowT-mass-4W_2302071732061_MwData.csv']
t1, T1, P1 = read_data(files)
T1, t1, P1 = lissage(np.array(T1), np.array(t1), np.array(P1), 51, 660.0)

files = ['lowT-mass-5W_2302071731344_MwData.csv']
t2, T2, P2 = read_data(files)
T2, t2, P2 = lissage(np.array(T2), np.array(t2), np.array(P2), 51, 496.0)

t_train = np.concatenate((t1, t2), axis=0)
T_train = np.concatenate((T1, T2), axis=0)
Q_train = np.concatenate((P1, P2), axis=0)

# # Inputs (interpolation)
# files = ['lowT-mass-4W_2302071732061_MwData.csv', 'lowT-mass-5W_2302071731344_MwData.csv']
# t_data, T_ir, P = read_data(files)

# t_train = np.array(t_data)
# T_train = np.array(T_ir)
# Q_train = np.array(P)

start = [0]
for i in range(len(t_train)-1):
    if (t_train[i+1] - t_train[i]) < 0:
        start.append(i+1)

step = 50
idx1_ = np.arange(start[0],start[1])
if idx1_[-1] not in idx1_[::step]:
    idx1 = np.append(idx1_[::step], idx1_[-1])
else:
    idx1 = idx1_[::step]
idx2_ = np.arange(start[1],len(t_train))
if idx2_[-1] not in idx2_[::step]:
   idx2 = np.append(idx2_[::step], idx2_[-1])
else:
    idx2 = idx2_[::step]
idx = idx1.tolist() + idx2.tolist()
idx_y0 = [start[1]-1] + [len(t_train)-1]
idx_CI = start

# kind = 'cubic'
# interp1 = interpolation(t_train[idx1], T_train[idx1], kind)
# interp2 = interpolation(t_train[idx2], T_train[idx2], kind)

# T_train = np.concatenate((interp1(t_train[idx1_]), 
#                           interp2(t_train[idx2_])), axis=0)

X = np.append(t_train.reshape(-1,1), T_train.reshape(-1,1), axis=1)
Y = Q_train.reshape(-1,1)

# Train data
X_train, Y_external = put_in_device(X, Y, device)

# Create PINN
f_hat = torch.zeros(X_train.shape[0],1).to(device)
PINN = Curiosity(X_train, Y_external, y0, idx, idx_y0, idx_CI, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

# Training
epochs = 100000
epoch = 1
vec_loss = []
while epoch <= epochs:
    PINN.optimizer.step(PINN.closure)
    vec_loss.append(float(PINN.loss(X_train, Y_external).detach().numpy()))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t total_loss: {PINN.total_loss:.2e} \t cTG_data_loss: {PINN.loss_cTG_data:.2e} \t T_ode_loss: {PINN.loss_T_ode:.2e} \t cTG_ode_loss: {PINN.loss_cTG_ode:.2e}')
    if epoch == 10000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)
    #if epoch == 10000:
    #    PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)
    #if epoch == 40000:
    #    PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-6)
    epoch += 1

PINN.save_model(PATH)

plt.plot(vec_loss)
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), X_train[idx1_,1].detach().numpy())
plt.plot(X_train[idx1,0].detach().numpy(), T_train[idx1], 'o')
plt.plot(X_train[idx2_,0].detach().numpy(), X_train[idx2_,1].detach().numpy())
plt.plot(X_train[idx2,0].detach().numpy(), T_train[idx2], 'o')
plt.xlabel('Time [sec]')
plt.ylabel(r'Temperature [$\degree$C]')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), PINN.PINN(X_train)[idx1_,0].detach().numpy(), label='Q1')
plt.plot(X_train[idx_CI+idx_y0,0].detach().numpy(), y0, 'o', label='Conditions')
plt.plot(X_train[idx2_,0].detach().numpy(), PINN.PINN(X_train)[idx2_,0].detach().numpy(), label='Q2')
plt.legend()
plt.xlabel('Time [sec]')
plt.ylabel(r'Concentration [mol/L]')
plt.show()

plt.plot(X_train[idx1_,1].detach().numpy(), PINN.PINN(X_train)[idx1_,1].detach().numpy(), 'o')
plt.plot(X_train[idx2_,1].detach().numpy(), PINN.PINN(X_train)[idx2_,1].detach().numpy(), 'o')
plt.xlabel(r'Temperature [$\degree$C]')
plt.ylabel(r'Kinetic constant [s$^{-1}$]')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), PINN.PINN(X_train)[idx1_,1].detach().numpy())
plt.plot(X_train[idx2_,0].detach().numpy(), PINN.PINN(X_train)[idx2_,1].detach().numpy())
plt.xlabel('Time [sec]')
plt.ylabel(r'Kinetic constant [s$^{-1}$]')
plt.show()