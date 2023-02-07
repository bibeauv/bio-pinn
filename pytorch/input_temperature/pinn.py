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
t_train, \
T_train, \
cTG_train, \
c_pred, idx = read_data(Ea, A, t1, t2, T1, T2, collocation_points, device)
c_train = [cTG_train]

# ODEs residual
f_hat = torch.zeros(t_train.shape[0],1).to(device)

# Sub PINNs
Ea_pred = 100.
A_pred = 10.
k_pred = [Ea_pred, A_pred]
PINN_cTG = cTGNN(device, f_hat, c_pred, k_pred, t_train, 1/T_train, cTG_train, idx)
PINNs = [PINN_cTG]

x_train = t_train

# Initial predictions
for i, PINN in enumerate(PINNs):
    c_pred[:,i] = PINN.dnn(x_train).detach().clone().flatten()
    PINN.pred(k_pred)

# Training
epoch = 0
max_epochs = 200000
vec_loss = []
while epoch < max_epochs:
    # Backward
    for i, PINN in enumerate(PINNs):
            train_cNN(PINN, i, c_pred, k_pred, t_train, 1/T_train)
            vec_loss.append(float(PINN.loss(x_train, c_train[i]).detach().numpy()))
    
    epoch += 1
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t cTG_loss: {vec_loss[-1]:.4e}')
    
    if epoch == 20000:
        for PINN in PINNs:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-3)
            
    if epoch == 100000:
        for PINN in PINNs:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)
            
    #if epoch == X:
    #    for PINN in PINNs:
    #        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)

print('\n')

plt.plot(vec_loss)
plt.xscale('log')
plt.yscale('log')
plt.show()

dt = t2 / (collocation_points - 1)
Ea_euler = PINN.dnn.Ea.detach().numpy()[0]
A_euler = PINN.dnn.A.detach().numpy()[0]
T_euler = T_train.detach().numpy()
t_num, y_num = euler_explicite([0.6], T_euler.flatten(), \
               Ea_euler, A_euler, dt, t2, [A_euler*np.exp(-Ea_euler/T_euler[0][0])])
t_num2, y_num2 = euler_explicite([0.6], T_euler.flatten(), \
                 Ea, A, dt, t2, [A*np.exp(-Ea/T_euler[0][0])])

fig, axs = plt.subplots(2)
axs[0].plot(t_train.detach().numpy(), PINNs[0].dnn(x_train).detach().numpy(), label='PINN')
axs[0].plot(t_num, y_num, '--', label='Euler')
axs[0].plot(t_num2, y_num2, label='Real')
axs[0].set_ylabel('Concentration [mol/L]')
axs[0].legend()
axs[1].plot(t_train.detach().numpy(), (1/T_train).detach().numpy())
axs[1].set_ylabel('Température [degC]')
axs[1].set_xlabel('Temps [min]')
plt.show()

print(PINN.dnn.Ea)
print(PINN.dnn.A)