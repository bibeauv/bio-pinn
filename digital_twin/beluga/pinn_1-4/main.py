from data import *
from pinn import *
import os

PATH = os.getcwd()

files = ['exp1_6W.csv', 'exp2_6W.csv', 'exp3_6W.csv',
         'exp1_5W.csv', 'exp2_5W.csv', 'exp3_5W.csv',
         'exp1_4W.csv', 'exp2_4W.csv', 'exp3_4W.csv']

X, Y, Z, idx, idx_y0, idx_yf = gather_data(files, 'T_train.csv')

device = torch.device('cuda')
X_train, Y_train, Z_train = put_in_device(X, Y, Z, device)
f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# Template
learning_rate = 1e-3
E = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
A = [1/240, 1/240, 1/240, 1/240, 1/240, 1/240]
neurons = 64
layers = 3
regularization = 5

class parameters():
    Q = 6
    m_Cp = 10
prm = parameters()

PINN = Curiosity(X_train, Y_train, Z_train,
                 idx, idx_y0, idx_yf, f_hat, learning_rate,
                 E, A,
                 neurons, regularization, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 100000
while epoch <= max_epochs:

    try:

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

        if epoch == 5000:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)

        if epoch == 25000:
            PINN.regularization = 50

        if epoch == 50000:
            PINN.regularization = 500

        if epoch == 75000:
            PINN.regularization = 5000

        if epoch == 100000:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-5)

        epoch += 1

    except KeyboardInterrupt:
        break
    
torch.save(PINN.PINN, PATH + '/model.pt')

with open('loss.txt', 'w') as f:
    f.write(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e}')
