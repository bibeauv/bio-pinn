from data import *
from pinn import *
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()

files = ['exp1.csv']

collocation_points = 181
X, Y, idx, idx_y0 = gather_data(files, collocation_points)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)
f_hat = f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# Template
learning_rate = 1e-3
E = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
A = [0.021309416741132736, 0.010142019018530846, 0.04338346794247627, 0.044596560299396515, 0.020075593143701553, 0.0007612420013174415]
e = 0.5
c1 = 0.1
c2 = 0.1
neurons = 10
regularization = 1000

class parameters():
    m_Cp = 10
    Q = 6
prm = parameters()

PINN = Curiosity(X_train, Y_train, idx, idx_y0, f_hat, learning_rate,
                 E, A, e, c1, c2,
                 neurons, regularization, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 1000000
while epoch <= max_epochs:

    PINN.optimizer.step(PINN.closure)

    PINN.PINN.E1.data.clamp_(min=0.)
    PINN.PINN.A1.data.clamp_(min=0.)
    PINN.PINN.E2.data.clamp_(min=0.)
    PINN.PINN.A2.data.clamp_(min=0.)
    PINN.PINN.E3.data.clamp_(min=0.)
    PINN.PINN.A3.data.clamp_(min=0.)
    PINN.PINN.E4.data.clamp_(min=0.)
    PINN.PINN.A4.data.clamp_(min=0.)
    PINN.PINN.E5.data.clamp_(min=0.)
    PINN.PINN.A5.data.clamp_(min=0.)
    PINN.PINN.E6.data.clamp_(min=0.)
    PINN.PINN.A6.data.clamp_(min=0.)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_T_data: {PINN.loss_T_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')

    if epoch == 100000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)

    if epoch == 500000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)
    
    epoch += 1
    
torch.save(PINN.PINN, PATH + '/model.pt')

with open('loss.txt', 'w') as f:
    f.write(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_T_data: {PINN.loss_T_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')
