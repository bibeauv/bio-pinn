# Importation de librairies
import numpy as np
import matplotlib.pyplot as plt

# Set seed
np.random.seed(1234)

def EDO(y, prm):
    
    k1 = prm[0]
    k2 = prm[1]
    k3 = prm[2]
    k4 = prm[3]
    k5 = prm[4]
    k6 = prm[5]
    
    f = np.zeros(5)
    
    f[1] = - k1*y[1] + k2*y[2]*y[0]
    f[2] = + k1*y[1] - k2*y[2]*y[0] - k3*y[2] + k4*y[3]*y[0]
    f[3] = + k3*y[2] - k4*y[3]*y[0] - k5*y[3] + k6*y[4]*y[0]
    f[4] = + k5*y[3] - k6*y[4]*y[0]
    f[0] = + k1*y[1] - k2*y[2]*y[0] + k3*y[2] - k4*y[3]*y[0] + k5*y[3] - k6*y[4]*y[0]
    
    return f

def euler_explicite(y0, dt, tf, prm):
    
    mat_y = np.array([y0])
    
    t = np.array([0])
    while t[-1] < tf:
        y = y0 + dt * EDO(y0, prm)
        
        mat_y = np.append(mat_y, [y], axis=0)
        t = np.append(t, t[-1]+dt)
        
        y0 = np.copy(y)
    
    return t, mat_y

def plot_loss(mat_loss):
    
    plt.plot(mat_loss[:,0], label='loss_cB')
    plt.plot(mat_loss[:,1], label='loss_cTG')
    plt.plot(mat_loss[:,2], label='loss_cDG')
    plt.plot(mat_loss[:,3], label='loss_cMG')
    plt.plot(mat_loss[:,4], label='loss_cG')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_pred(t_train, PINNs, idx, t_euler, y_euler, c_train):
    
    fig, axs = plt.subplots(2,2)

    # Biodiesel
    axs[0,0].plot(t_train.detach().numpy(), PINNs[0].dnn(t_train).detach().numpy())
    axs[0,0].plot(t_euler, y_euler[:,0], '--')
    axs[0,0].set_title('ME')
    axs[0,0].set_xlabel('Time [min]')
    axs[0,0].set_ylabel('Concentration [mol/L]')

    # TG
    axs[0,1].plot(t_train.detach().numpy(), PINNs[1].dnn(t_train).detach().numpy())
    axs[0,1].plot(t_train.detach().numpy()[idx], c_train[1][idx], 'o')
    axs[0,1].plot(t_euler, y_euler[:,1], '--')
    axs[0,1].set_title('TG')
    axs[0,1].set_xlabel('Time [min]')
    axs[0,1].set_ylabel('Concentration [mol/L]')

    # DG
    axs[1,0].plot(t_train.detach().numpy(), PINNs[2].dnn(t_train).detach().numpy())
    axs[1,0].plot(t_train.detach().numpy()[idx], c_train[2][idx], 'o')
    axs[1,0].plot(t_euler, y_euler[:,2], '--')
    axs[1,0].set_title('DG')
    axs[1,0].set_xlabel('Time [min]')
    axs[1,0].set_ylabel('Concentration [mol/L]')

    # MG
    axs[1,1].plot(t_train.detach().numpy(), PINNs[3].dnn(t_train).detach().numpy())
    axs[1,1].plot(t_train.detach().numpy()[idx], c_train[3][idx], 'o')
    axs[1,1].plot(t_euler, y_euler[:,3], '--')
    axs[1,1].set_title('MG')
    axs[1,1].set_xlabel('Time [min]')
    axs[1,1].set_ylabel('Concentration [mol/L]')
    
    plt.show()
