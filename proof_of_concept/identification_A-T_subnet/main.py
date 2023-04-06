from data import *
from pinn import *
import matplotlib.pyplot as plt

y0 = np.array([1., 35.])
t1 = 0
t2 = 300
collocation_points = 301

class parameters():
    k1 = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = 0.2
    c1 = -0.01
    c2 = 0.1
    A = 0.1
    E = 100

prm = parameters()

X, Y = create_data(y0, t1, t2, collocation_points, prm)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)

idx = np.arange(0, collocation_points)
idx = idx[::50]
idx = idx.tolist()

f_hat = torch.zeros(X_train.shape[0], 1).to(device)

PINNA = subnetA(X_train, Y_train, idx, f_hat, device, prm)
PINNT = subnetT(X_train, Y_train, idx, f_hat, device, prm)

PINNA.add_other_PINN(PINNT)
PINNT.add_other_PINN(PINNA)

PINNs = [PINNA, PINNT]

# Make all outputs positive
PINNA.c = PINNA.PINN(X_train).detach()
PINNT.T = PINNT.PINN(X_train).detach()
for i, p in enumerate(PINNA.PINN.parameters()):
    p.data.clamp_(min=0.)
for i, p in enumerate(PINNT.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 70000
vec_loss_PINNA = []
vec_loss_PINNT = []
while epoch <= max_epochs:
    
    PINNA.optimizer.step(PINNA.closure)
    PINNA.A.data.clamp_(min=0.)
    PINNA.E.data.clamp_(min=0.)

    PINNT.optimizer.step(PINNT.closure)

    vec_loss_PINNA.append(PINNA.total_loss.detach().numpy())
    vec_loss_PINNT.append(PINNT.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_cA_data: {PINNA.loss_c_data:.4e} \t loss_cA_ode: {PINNA.loss_c_ode:.4e} \t loss_T_data: {PINNT.loss_T_data:.4e} \t loss_T_ode: {PINNT.loss_T_ode:.4e}')

    if epoch == 50000:
        PINNT.optimizer = torch.optim.Adam(PINNT.params, lr=1e-4)
    
    epoch += 1

print(f'E = {float(PINNA.E.detach().numpy()):0.4f}')
print(f'A = {float(PINNA.A.detach().numpy()):0.4f}')
print(f'e = {float(PINNT.e.detach().numpy()):0.4f}')
print(f'c1 = {float(PINNT.c1.detach().numpy()):0.4f}')
print(f'c2 = {float(PINNT.c2.detach().numpy()):0.4f}')

plt.plot(vec_loss_PINNA, label='PINNA')
plt.plot(vec_loss_PINNT, label='PINNT')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

t = X_train[:,0].detach().numpy()
T = Y_train[:,1].detach().numpy()

cA_true = Y_train[:,0].detach().numpy()
T_true = Y_train[:,1].detach().numpy()

cA_pred = PINNA.PINN(X_train).detach().numpy()
T_pred = PINNT.PINN(X_train).detach().numpy()

k1_pred = float(PINNA.A.detach().numpy()) * np.exp(-float(PINNA.E.detach().numpy()) / T_pred)
k1_true = prm.A * np.exp(-prm.E / T)

class parameters2():
    k1 = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = float(PINNT.e.detach().numpy())
    c1 = float(PINNT.c1.detach().numpy())
    c2 = float(PINNT.c2.detach().numpy())
    A = float(PINNA.A.detach().numpy())
    E = float(PINNA.E.detach().numpy())

prm2 = parameters2()
X_test, Y_test = create_data(y0, t1, t2, collocation_points, prm2)

plt.plot(t[idx], cA_true[idx], 'o', label='cA_true')
plt.plot(X_test.flatten(), Y_test[:,0], '--', label='cA_test')
plt.plot(t, cA_pred, label='cA_pred')
plt.xlabel('Temps [sec]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t[idx], T_true[idx], 'o', label='T_true')
plt.plot(t, T_pred, label='T_pred')
plt.plot(X_test.flatten(), Y_test[:,1], '--', label='T_test')
plt.legend()
plt.show()

plt.plot(T_pred, k1_pred, label='k_pred')
plt.plot(T_true, k1_true, '--', label='k_true')
plt.legend()
plt.show()
