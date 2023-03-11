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

# Initial loss
mat_loss = np.zeros((1,5))
for i, PINN in enumerate(PINNs):
        PINN.pred(c_pred, k_pred)
        mat_loss[:,i] = float(PINN.loss(t_train, c_train[i]).detach().numpy())

# Training
epoch = 0
max_epochs = 10000
while epoch < max_epochs:
    vec_loss = np.zeros((1,5))
    # Backward
    for i, PINN in enumerate(PINNs):
            train_cNN(PINN, i, c_pred, k_pred, t_train)
            vec_loss[0][i] = float(PINN.loss(t_train, c_train[i]).detach().numpy())
    
    mat_loss = np.append(mat_loss, vec_loss, axis=0)
    mat_k_pred = np.append(mat_k_pred, [k_pred], axis=0)
    
    epoch += 1

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, \t cB_loss: {vec_loss[0][0]:.4e} \t cTG_loss: {vec_loss[0][1]:.4e} \t cDG_loss: {vec_loss[0][2]:.4e} \t cMG_loss: {vec_loss[0][3]:.4e} \t cG_loss: {vec_loss[0][4]:.4e}')
        
    if epoch == 8000:
        for PINN in PINNs:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-3)

    if epoch == 15000:
        for PINN in PINNs:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)

print('\n')

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

print('k_pred: ', k_pred)

# Euler
y0 = np.array([0,0.570199408,0.063624505,0,0])
dt = 0.001
tf = 8
prm = []
for i in range(1,7):
    prm.append(kinetics[i][-1])
t_euler, y_euler = euler_explicite(y0, dt, tf, prm)

# Loss plot
plot_loss(mat_loss)

# Plot
plot_pred(t_train, PINNs, idx, t_euler, y_euler, c_train)
