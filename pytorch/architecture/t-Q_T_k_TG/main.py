from data import *
from pinn import *
import numpy as np
import matplotlib.pyplot as plt

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
for Q in [0.3,0.5]:
    prm.Q = Q
    X_, Y_ = create_data(y0, t1, t2, collocation_points, prm)
    Q_ = np.ones(X_.shape).reshape(-1,1)*prm.Q
    X_power = np.append(X_power, Q_, axis=0)
    X = np.append(X, X_, axis=0)
    Y = np.append(Y, Y_, axis=0)

t_data = np.linspace(t1, t2, data_points)
idx = find_idx(t_data, X)

# Train data
X = np.append(X, X_power, axis=1)
X_train, Y_train = put_in_device(X, Y, device)

# Create PINN
f_hat = torch.zeros(X_train.shape[0],1).to(device)
PINN_TG = TGNeuralNet(X_train, Y_train, idx, f_hat, device)

# Training
epochs = 50000
epoch = 1
vec_loss = []
while epoch < epochs:
    PINN_TG.optimizer.step(PINN_TG.closure)
    vec_loss.append(PINN_TG.loss(X_train, Y_train).detach().numpy())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t loss: {vec_loss[-1]:.4e}')
    if epoch == 7500:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-5)
    if epoch == 11000:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-6)
    if epoch == 30000:
        PINN_TG.optimizer = torch.optim.Adam(PINN_TG.params, lr=1e-8)
    epoch += 1

plt.plot(vec_loss)
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.show()

plt.plot(X_train[0:101,0].detach().numpy(), PINN_TG.PINN(X_train[0:101]).detach().numpy())
plt.plot(X_train[idx[:-1:3],0].detach().numpy(), Y_train[idx[:-1:3],0].detach().numpy(), 'o')
plt.show()