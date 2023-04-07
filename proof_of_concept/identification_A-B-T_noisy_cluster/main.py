from data import *
from pinn import *

PATH = os.getcwd() + '/model.pt'

y0 = np.array([1., 0., 35.])
t1 = 0
t2 = 300
collocation_points = 301

class parameters():
    k1 = None
    k2 = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = 0.2
    c1 = -0.01
    c2 = 0.1
    A1 = 0.15
    E1 = 150
    A2 = 0.2
    E2 = 200

prm = parameters()

X, Y, Y_no_noise = create_data(y0, t1, t2, collocation_points, prm, error_percentage=10/100)

device = torch.device('cuda')
X_train, Y_train = put_in_device(X, Y, device)

idx = np.arange(0, collocation_points)
idx = idx[::50]
idx = idx.tolist()

f_hat = torch.zeros(X_train.shape[0], 1).to(device)

PINN = Curiosity(X_train, Y_train, idx, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 100000
vec_loss = []
check = False
while epoch <= max_epochs:

    PINN.optimizer.step(PINN.closure)

    PINN.PINN.E1.data.clamp_(min=0.)
    PINN.PINN.A1.data.clamp_(min=0.)
    PINN.PINN.E2.data.clamp_(min=0.)
    PINN.PINN.A2.data.clamp_(min=0.)
    
    if epoch % 10000 == 0:
    	print(f'Epoch {epoch} \t loss_data: {PINN.loss_data:.4e} \t loss_cA_ode: {PINN.loss_cA_ode:.4e} \t loss_cB_ode: {PINN.loss_cB_ode:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')

    epoch += 1

print(f'E1 = {float(PINN.PINN.E1.detach().cpu().numpy()):.4f}')
print(f'A1 = {float(PINN.PINN.A1.detach().cpu().numpy()):.4f}')
print(f'E2 = {float(PINN.PINN.E2.detach().cpu().numpy()):.4f}')
print(f'A2 = {float(PINN.PINN.A2.detach().cpu().numpy()):.4f}')
print(f'e = {float(PINN.PINN.e.detach().cpu().numpy()):.4f}')
print(f'c1 = {float(PINN.PINN.c1.detach().cpu().numpy()):.4f}')
print(f'c2 = {float(PINN.PINN.c2.detach().cpu().numpy()):.4f}')

PINN.save_model(PATH)

