from data import *
from pinn import *
import os

PATH = os.getcwd()

GC_files = ['GC_file1.csv', 'GC_file2.csv', 'GC_file3.csv']
MW_files = ['MW_file1.csv', 'MW_file2.csv', 'MW_file3.csv']

X, Y, idx = gather_data(GC_files, MW_files)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)
f_hat = f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# Template
learning_rate = 1e-3
E = {{E}}
A = {{A}}
e = {{e}}
c1 = {{c1}}
c2 = {{c2}}
neurons = {{neurons}}
regularization = {{regularization}}

class parameters():
    m_Cp = 10
    Q = 6
    dHrx1 = -10000
    dHrx2 = 10000
    dHrx3 = -10000
    V = 6.3 / 1000
prm = parameters()

PINN = Curiosity(X_train, Y_train, idx, f_hat, learning_rate, E, A, e, c1, c2, neurons, device, prm, regularization=1)

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
        print(f'Epoch {epoch} \t loss_data: {PINN.loss_data:.4e} \t loss_ode: {PINN.loss_ode:.4e}')

    if epoch == 20000:
        PINN.regularization = regularization

    epoch += 1
    
PINN.save_model(PATH)