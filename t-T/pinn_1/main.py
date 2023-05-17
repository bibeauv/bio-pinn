from data import *
from pinn import *
import os

PATH = os.getcwd()

files = ['T_train.csv']

X, Y = gather_data(files)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)
f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# Template
learning_rate = 1e-3
e = 0.5
c1 = 0.1
c2 = 0.1
neurons = 10
regularization = 1

class parameters():
    m_Cp = 10
prm = parameters()

PINN = Curiosity(X_train, Y_train, f_hat, learning_rate,
                 e, c1, c2,
                 neurons, regularization, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 100000
while epoch <= max_epochs:

    try:
        PINN.optimizer.step(PINN.closure)

        if epoch % 1000 == 0:
            print(f'Epoch {epoch} \t loss_T_data: {PINN.loss_T_data:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')

        if epoch == 25000:
            PINN.regularization = 10
            PINN.optimzer = torch.optim.Adam(PINN.params, lr=1e-4)
            
        if epoch == 50000:
            PINN.regularization = 100
            PINN.optimzer = torch.optim.Adam(PINN.params, lr=1e-5)

        if epoch == 75000:
            PINN.regularization = 1000

    except KeyboardInterrupt:
        break
    
    epoch += 1
    
torch.save(PINN.PINN, PATH + '/model.pt')

with open('loss.txt', 'w') as f:
    f.write(f'Epoch {epoch} \t loss_T_data: {PINN.loss_T_data:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')