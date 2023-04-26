from data import *
from pinn import *
import os

PATH = os.getcwd()

files = ['exp1.csv', 'exp2.csv', 'exp3.csv']

X, Y, Z, idx, idx_y0 = gather_data(files, 'T_train.csv', 5.0)

device = torch.device('cpu')
X_train, Y_train, Z_train = put_in_device(X, Y, Z, device)
f_hat = f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# Template
learning_rate = 1e-3
E = [97.5, 91.9, 103.9, 82.3, 0., 0.]
A = [0.38, 0.17, 0.54, 0.22, 0.036, 0.0014]
neurons = 10
regularization = 5000

class parameters():
    Q = 5
prm = parameters()

PINN = Curiosity(X_train, Y_train, Z_train, idx, idx_y0, f_hat, learning_rate,
                 E, A,
                 neurons, regularization, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 100000
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
        print(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e}')

    if epoch == 10000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)

    if epoch == 20000:
        PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)
        
    # if PINN.loss_c_ode <= 1e-8:
    #     break

    epoch += 1
    
torch.save(PINN.PINN, PATH + '/model.pt')

with open('loss.txt', 'w') as f:
    f.write(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e}')