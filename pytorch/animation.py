from preprocess import *
from source import *
from postprocess import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
collocation_points = 100 * 8 + 1
t_train, \
cB_train, \
cTG_train, \
cDG_train, \
cMG_train, \
cG_train, \
c_pred, idx = read_data('bio.csv', device, 0, 8, collocation_points)
c_train = [cB_train, cTG_train, cDG_train, cMG_train, cG_train]

# ODEs residual
f_hat = torch.zeros(t_train.shape[0],1).to(device)

# Sub PINNs
PINN_cTG = cTGNN(device, f_hat, c_pred, k_pred, t_train, cTG_train, idx)
PINN_cDG = cDGNN(device, f_hat, c_pred, k_pred, t_train, cDG_train, idx)
PINN_cMG = cMGNN(device, f_hat, c_pred, k_pred, t_train, cMG_train, idx)
PINN_cG = cGNN(device, f_hat, c_pred, k_pred, t_train, cG_train)
PINN_cB = cBNN(device, f_hat, c_pred, k_pred, t_train, cB_train)
PINNs = [PINN_cB, PINN_cTG, PINN_cDG, PINN_cMG, PINN_cG]

# Initial predictions
for i, PINN in enumerate(PINNs):
    c_pred[:,i] = PINN.dnn(t_train).detach().clone().flatten()

# Animation
fig, axs = plt.subplots(2,2)
axs[0,0].set_ylim([-0.05,1.8])
axs[0,1].set_ylim([-0.02,0.6])
axs[1,0].set_ylim([-0.02,0.3])
axs[1,1].set_ylim([-0.02,0.2])
for ax in axs:
    for a in ax:
        a.set_xlim([-0.5,8.5])

line1, = axs[0,0].plot([], [])
line2, = axs[0,0].plot([], [], '--')
title = axs[0,0].text(1.0,2.0, "")

# TG
line3, = axs[0,1].plot([], [])
line4, = axs[0,1].plot([], [], 'o')
line5, = axs[0,1].plot([], [], '--')

# DG
line6, = axs[1,0].plot([], [])
line7, = axs[1,0].plot([], [], 'o')
line8, = axs[1,0].plot([], [], '--')

# MG
line9, = axs[1,1].plot([], [])
line10, = axs[1,1].plot([], [], 'o')
line11, = axs[1,1].plot([], [], '--')

max_epochs = 5000
def make_animation(j):
    epoch = j
    while epoch < max_epochs:
        # Backward
        for i, PINN in enumerate(PINNs):
                train_cNN(PINN, i, c_pred, k_pred, t_train)
        
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

        # Euler
        y0 = np.array([0,0.491115698,0.030281241,0,0])
        dt = 0.001
        tf = 8
        prm = []
        for i in range(1,7):
            prm.append(kinetics[i][-1])
        t_euler, y_euler = euler_explicite(y0, dt, tf, prm)

        # Plot
        # Biodiesel
        line1.set_data(t_train.detach().numpy(), PINNs[0].dnn(t_train).detach().numpy())
        line2.set_data(t_euler, y_euler[:,0])

        # TG
        line3.set_data(t_train.detach().numpy(), PINNs[1].dnn(t_train).detach().numpy())
        line4.set_data(t_train.detach().numpy()[idx], c_train[1][idx])
        line5.set_data(t_euler, y_euler[:,1])

        # DG
        line6.set_data(t_train.detach().numpy(), PINNs[2].dnn(t_train).detach().numpy())
        line7.set_data(t_train.detach().numpy()[idx], c_train[2][idx])
        line8.set_data(t_euler, y_euler[:,2])

        # MG
        line9.set_data(t_train.detach().numpy(), PINNs[3].dnn(t_train).detach().numpy())
        line10.set_data(t_train.detach().numpy()[idx], c_train[3][idx])
        line11.set_data(t_euler, y_euler[:,3])

        title.set_text(f'Epoch: {j}')

        epoch += 1

        if epoch == 1000:
            for PINN in PINNs:
                PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-3)
        
        print(epoch, end='\r')

        return line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11

myAnimation = animation.FuncAnimation(fig, make_animation, frames=max_epochs)
writergif = animation.PillowWriter(fps=60)
myAnimation.save('animation.gif', writer=writergif)