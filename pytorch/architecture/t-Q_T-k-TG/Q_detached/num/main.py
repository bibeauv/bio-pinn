from data import *
from pinn import *
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.getcwd() + '/model.pt'

# Data parameters
y0 = [0.6, 35]
t1 = 0
t2 = 10
collocation_points = 101
data_points = 6
device = torch.device('cpu')

# Parameters
class parameters():
    Ea = 200                # J/mol
    A = 50                  # -
    dHrx = -40              # J/mol
    Q = None                # J/s
    V = 1                   # L
    k = A*np.exp(-Ea/y0[1]) # s^{-1}
prm = parameters()

# Inputs
prm.Q = 0.2
X, Y = create_data(y0, t1, t2, collocation_points, prm)
X_power = np.ones(X.shape).reshape(-1,1)*prm.Q
for Q in [0.2,0.2]:
    prm.Q = Q
    prm.k = prm.A*np.exp(-prm.Ea/y0[1])
    X_, Y_ = create_data(y0, t1, t2, collocation_points, prm)
    Q_ = np.ones(X_.shape).reshape(-1,1)*prm.Q
    X_power = np.append(X_power, Q_, axis=0)
    X = np.append(X, X_, axis=0)
    Y = np.append(Y, Y_, axis=0)

t_data = np.linspace(t1, t2, data_points)
idx = find_idx(t_data, X)

# Real k
real_k = prm.A*np.exp(-prm.Ea/Y[:,1])

# Train data
X = np.append(X, X_power, axis=1)
X_train, Y_train = put_in_device(X, Y, device)
x_train = X_train[:,0].reshape(-1,1)
Q_train = X_train[:,1].reshape(-1,1)

# Create PINN
f_hat = torch.zeros(x_train.shape[0],1).to(device)
PINN_TG = Curiosity(x_train, Y_train, Q_train, idx, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN_TG.PINN.parameters()):
    p.data.clamp_(min=0.)

# Training
epochs = 50000
epoch = 1
vec_loss = []
while epoch <= epochs:
    PINN_TG.optimizer.step(PINN_TG.closure)
    vec_loss.append(float(PINN_TG.loss(x_train, Y_train).detach().numpy()))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t loss: {vec_loss[-1]:.4e}')
    if epoch == 5000:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-3)
    if epoch == 10000:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-4)
    if epoch == 30000:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-5)
    epoch += 1

PINN_TG.save_model(PATH)
    
plt.plot(vec_loss)
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.show()

plt.plot(X_train[0:101,0].detach().numpy(), PINN_TG.PINN(x_train[0:101])[:,0].detach().numpy())
plt.plot(X_train[idx[::3],0].detach().numpy(), Y_train[idx[::3],0].detach().numpy(), 'o')
plt.plot(X_train[101:202,0].detach().numpy(), PINN_TG.PINN(x_train[101:202])[:,0].detach().numpy())
plt.plot(X_train[idx[1::3],0].detach().numpy(), Y_train[idx[1::3],0].detach().numpy(), 'o')
plt.plot(X_train[202:303,0].detach().numpy(), PINN_TG.PINN(x_train[202:303])[:,0].detach().numpy())
plt.plot(X_train[idx[2::3],0].detach().numpy(), Y_train[idx[2::3],0].detach().numpy(), 'o')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.show()

plt.plot(X_train[0:101,0].detach().numpy(), PINN_TG.PINN(x_train[0:101])[:,1].detach().numpy())
plt.plot(X_train[idx[::3],0].detach().numpy(), Y_train[idx[::3],1].detach().numpy(), 'o')
plt.plot(X_train[101:202,0].detach().numpy(), PINN_TG.PINN(x_train[101:202])[:,1].detach().numpy())
plt.plot(X_train[idx[1::3],0].detach().numpy(), Y_train[idx[1::3],1].detach().numpy(), 'o')
plt.plot(X_train[202:303,0].detach().numpy(), PINN_TG.PINN(x_train[202:303])[:,1].detach().numpy())
plt.plot(X_train[idx[2::3],0].detach().numpy(), Y_train[idx[2::3],1].detach().numpy(), 'o')
plt.xlabel('Time [min]')
plt.ylabel(r'Temperature [$\degree$C]')
plt.show()

plt.plot(Y_train[0:101,1].detach().numpy(), PINN_TG.PINN(x_train[0:101])[:,2].detach().numpy())
plt.plot(Y_train[0:101,1].detach().numpy(), real_k[0:101], '--')
plt.plot(Y_train[101:202,1].detach().numpy(), PINN_TG.PINN(x_train[101:202])[:,2].detach().numpy())
plt.plot(Y_train[101:202,1].detach().numpy(), real_k[101:202], '--')
plt.plot(Y_train[202:303,1].detach().numpy(), PINN_TG.PINN(x_train[202:303])[:,2].detach().numpy())
plt.plot(Y_train[202:303,1].detach().numpy(), real_k[202:303], '--')
plt.xlabel(r'Temperature [$\degree$C]')
plt.ylabel(r'Kinetic constant [min$^{-1}$]')
plt.show()