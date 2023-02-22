from data import *
from pinn import *
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.getcwd() + '/model.pt'

# Data parameters
device = torch.device('cpu')
y0 = torch.from_numpy(np.array([0.6, 0.6, 0.6,
                                0.496, 0.468, 0.028]).reshape(-1,1)).float().to(device)

# Parameters
class parameters():
    dHrx = -45650              # J/mol
    epsilon = 0.1              # -
    V = 6.3                    # mL
    m = 5.6                    # g
    Cp = 2                     # J/g/C
prm = parameters()

# Inputs
files = ['5W-2min_2301181635123_MwData.csv', '5W-4min_2301181635393_MwData.csv', '5W-8min_2301181635551_MwData.csv']
t_data, T_ir, P = read_data(files)

t_train = np.array(t_data)
T_train = np.array(T_ir)
Q_train = np.array(P)

start = [0]
for i in range(len(t_train)-1):
    if (t_train[i+1] - t_train[i]) < 0:
        start.append(i+1)
idx1_ = np.arange(start[0],start[1])
idx1 = idx1_[::10]
idx2_ = np.arange(start[1],start[2])
idx2 = idx2_[::10]
idx3_ = np.arange(start[2],len(t_train))
idx3 = idx3_[::10]
idx = idx1.tolist() + idx2.tolist() + idx3.tolist()
idx_y0 = start + [start[1]-1] + [start[2]-1] + [len(t_train)-1]

X = np.append(t_train.reshape(-1,1), Q_train.reshape(-1,1), axis=1)
Y = T_train.reshape(-1,1)

# Train data
x_train = torch.from_numpy(t_train.reshape(-1,1)).float().to(device)
X_train, Y_train = put_in_device(X, Y, device)

# Create PINN
f_hat = torch.zeros(X_train.shape[0],1).to(device)
Curiosity = Curiosity(x_train, Y_train, y0, idx, idx_y0, f_hat, device, prm)
TheRover = TheRover(Curiosity, X_train, Y_train, y0, idx, idx_y0, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(Curiosity.PINN.parameters()):
    p.data.clamp_(min=0.)
for i, p in enumerate(TheRover.PINN.parameters()):
    p.data.clamp_(min=0.)

# Training
epochs = 50000
epoch = 1
vec_loss_Curiosity = []
vec_loss_TheRover = []
while epoch <= epochs:
    Curiosity.optimizer.step(Curiosity.closure)
    TheRover.optimizer.step(TheRover.closure)
    vec_loss_Curiosity.append(float(Curiosity.loss(x_train, Y_train).detach().numpy()))
    vec_loss_TheRover.append(float(TheRover.loss(X_train, Y_train).detach().numpy()))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}')
    if epoch == 2000:
        Curiosity.optimizer = torch.optim.Adam(Curiosity.params, lr=1e-5)
    if epoch == 20000:
        Curiosity.optimizer = torch.optim.Adam(Curiosity.params, lr=0.5e-5)
    epoch += 1

Curiosity.save_model(PATH)

plt.plot(vec_loss_Curiosity, label='Curiosity')
plt.plot(vec_loss_TheRover, label='TheRover')
plt.legend()
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), TheRover.PINN(X_train)[idx1_,0].detach().numpy())
plt.plot(X_train[idx1,0].detach().numpy(), T_train[idx1], 'o')
plt.plot(X_train[idx2_,0].detach().numpy(), TheRover.PINN(X_train)[idx2_,0].detach().numpy())
plt.plot(X_train[idx2,0].detach().numpy(), T_train[idx2], 'o')
plt.plot(X_train[idx3_,0].detach().numpy(), TheRover.PINN(X_train)[idx3_,0].detach().numpy())
plt.plot(X_train[idx3,0].detach().numpy(), T_train[idx3], 'o')
plt.xlabel('Time [sec]')
plt.ylabel(r'Temperature [$\degree$C]')
plt.show()

plt.plot(X_train[idx1_,0].detach().numpy(), Curiosity.PINN(x_train)[idx1_,0].detach().numpy(), label='Q1')
plt.plot(X_train[idx_y0,0].detach().numpy(), y0, 'o', label='Conditions')
plt.plot(X_train[idx2_,0].detach().numpy(), Curiosity.PINN(x_train)[idx2_,0].detach().numpy(), label='Q2')
plt.plot(X_train[idx3_,0].detach().numpy(), Curiosity.PINN(x_train)[idx3_,0].detach().numpy(), label='Q3')
plt.legend()
plt.xlabel('Time [sec]')
plt.ylabel(r'Concentration [mol/L]')
plt.show()

plt.plot(T_train[idx1], Curiosity.PINN(x_train)[idx1,1].detach().numpy(), 'o')
plt.plot(T_train[idx2], Curiosity.PINN(x_train)[idx2,1].detach().numpy(), 'o')
plt.plot(T_train[idx3], Curiosity.PINN(x_train)[idx3,1].detach().numpy(), 'o')
plt.xlabel(r'Temperature [$\degree$C]')
plt.ylabel(r'Kinetic constant [s$^{-1}$]')
plt.show()