from preprocess import *
from source import *
from postprocess import *
import matplotlib.pyplot as plt

# Create device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kinetic constants
k1 = 1.
k2 = 1.
k3 = 1.
k4 = 1.
k5 = 1.
k6 = 1.
k_pred = [k1,k2,k3,k4,k5,k6]
mat_k_pred = np.array([k_pred])

# Read and prepare data
t_train, cB_train, cTG_train, cDG_train, cMG_train, cG_train, c_pred = read_data('bio.csv', device)

# ODEs
f_hat = torch.zeros(t_train.shape[0],1).to(device)
alpha = 0.99

# Sub PINNs
PINN_cTG = cTGNN(device, alpha, f_hat, c_pred, k_pred, t_train, cTG_train)
PINN_cDG = cDGNN(device, alpha, f_hat, c_pred, k_pred, t_train, cDG_train)
PINN_cMG = cMGNN(device, alpha, f_hat, c_pred, k_pred, t_train, cMG_train)
PINN_cG = cGNN(device, alpha, f_hat, c_pred, k_pred, t_train)
PINN_cB = cBNN(device, alpha, f_hat, c_pred, k_pred, t_train)

# Initial predictions
c_pred[:,0] = PINN_cB.dnn(t_train).detach().clone().flatten()
c_pred[:,1] = PINN_cTG.dnn(t_train).detach().clone().flatten()
c_pred[:,2] = PINN_cDG.dnn(t_train).detach().clone().flatten()
c_pred[:,3] = PINN_cMG.dnn(t_train).detach().clone().flatten()
c_pred[:,4] = PINN_cG.dnn(t_train).detach().clone().flatten()

# Initial loss
mat_loss = np.zeros((1,5))
PINN_cB.pred(c_pred, k_pred)
mat_loss[:,0] = float(PINN_cB.loss(t_train).detach().numpy())
PINN_cTG.pred(c_pred, k_pred)
mat_loss[:,1] = float(PINN_cTG.loss(t_train, cTG_train).detach().numpy())
PINN_cDG.pred(c_pred, k_pred)
mat_loss[:,2] = float(PINN_cDG.loss(t_train, cDG_train).detach().numpy())
PINN_cMG.pred(c_pred, k_pred)
mat_loss[:,3] = float(PINN_cMG.loss(t_train, cMG_train).detach().numpy())
PINN_cG.pred(c_pred, k_pred)
mat_loss[:,4] = float(PINN_cG.loss(t_train).detach().numpy())

# Training
epoch = 0
max_epochs = 200
while epoch < max_epochs:
    vec_loss = np.zeros((1,5))
    # Backward
    train_cNN(PINN_cB, 0, c_pred, k_pred, t_train)
    vec_loss[0][0] = float(PINN_cB.loss(t_train).detach().numpy())
    train_cNN(PINN_cTG, 1, c_pred, k_pred, t_train)
    vec_loss[0][1] = float(PINN_cTG.loss(t_train, cTG_train).detach().numpy())
    train_cNN(PINN_cDG, 2, c_pred, k_pred, t_train)
    vec_loss[0][2] = float(PINN_cDG.loss(t_train, cDG_train).detach().numpy())
    train_cNN(PINN_cMG, 3, c_pred, k_pred, t_train)
    vec_loss[0][3] = float(PINN_cMG.loss(t_train, cMG_train).detach().numpy())
    train_cNN(PINN_cG, 4, c_pred, k_pred, t_train)
    vec_loss[0][4] = float(PINN_cG.loss(t_train).detach().numpy())
    
    mat_loss = np.append(mat_loss, vec_loss, axis=0)
    mat_k_pred = np.append(mat_k_pred, [k_pred], axis=0)
    
    epoch += 1
    print(epoch, end='\r')
print('\n')

PINNs = [PINN_cB, PINN_cTG, PINN_cDG, PINN_cMG, PINN_cG]

# Kinetics
kinetics = {1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[]}

for PINN in PINNs:
    for i in range(1,7):
        kinetics[i].append(float(PINN.params[i-1][0].detach().numpy()))

print(k_pred)

# Euler
y0 = np.array([0,0.540121748,0.057018273,0,0])
dt = 0.001
tf = 6
prm = []
for i in range(1,7):
    prm.append(kinetics[i][-1])
t_euler, y_euler = euler_explicite(y0, dt, tf, prm)

# Loss plot
plt.plot(mat_loss[:,0], label='loss_cB')
plt.plot(mat_loss[:,1], label='loss_cTG')
plt.plot(mat_loss[:,2], label='loss_cDG')
plt.plot(mat_loss[:,3], label='loss_cMG')
plt.plot(mat_loss[:,4], label='loss_cG')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# Plot
fig, axs = plt.subplots(2,2)

# Biodiesel
axs[0,0].plot(t_train.detach().numpy(), PINN_cB.dnn(t_train).detach().numpy())
axs[0,0].plot(t_euler, y_euler[:,0], '--')
axs[0,0].set_title('ME')

# TG
axs[0,1].plot(t_train.detach().numpy(), PINN_cTG.dnn(t_train).detach().numpy())
axs[0,1].plot(t_train.detach().numpy(), cTG_train, 'o')
axs[0,1].plot(t_euler, y_euler[:,1], '--')
axs[0,1].set_title('TG')

# DG
axs[1,0].plot(t_train.detach().numpy(), PINN_cDG.dnn(t_train).detach().numpy())
axs[1,0].plot(t_train.detach().numpy(), cDG_train, 'o')
axs[1,0].plot(t_euler, y_euler[:,2], '--')
axs[1,0].set_title('DG')

# MG
axs[1,1].plot(t_train.detach().numpy(), PINN_cMG.dnn(t_train).detach().numpy())
axs[1,1].plot(t_train.detach().numpy(), cMG_train, 'o')
axs[1,1].plot(t_euler, y_euler[:,3], '--')
axs[1,1].set_title('MG')
plt.show()
