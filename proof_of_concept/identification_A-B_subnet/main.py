from data import *
from pinn import *
import matplotlib.pyplot as plt

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

X, Y = create_data(y0, t1, t2, collocation_points, prm)

device = torch.device('cpu')
X_train, Y_train = put_in_device(X, Y, device)

idx = np.arange(0, collocation_points)
idx = idx[::50]
idx = idx.tolist()

f_hat = torch.zeros(X_train.shape[0], 1).to(device)

kinetics = [75., 1., 215., 1.]

PINNA = subnetA(X_train, Y_train, idx, f_hat, device, prm, kinetics)
PINNB = subnetB(X_train, Y_train, idx, f_hat, device, prm, kinetics)

PINNA.add_other_PINN(PINNB)
PINNB.add_other_PINN(PINNA)

PINNs = [PINNA, PINNB]

# Make all outputs positive
for PINN in PINNs:
    PINN.c = PINN.PINN(X_train).detach()
    for i, p in enumerate(PINN.PINN.parameters()):
        p.data.clamp_(min=0.)

epoch = 0
max_epochs = 100000
vec_loss_PINNA = []
vec_loss_PINNB = []
while epoch <= max_epochs:
        
    PINNA.E1.data.fill_(float(PINNB.E1.detach().numpy()))
    PINNA.A1.data.fill_(float(PINNB.A1.detach().numpy()))
    PINNA.E2.data.fill_(float(PINNB.E2.detach().numpy()))
    PINNA.A2.data.fill_(float(PINNB.A2.detach().numpy()))
    PINNA.optimizer.step(PINNA.closure)
    PINNA.E1.data.clamp_(min=0.)
    PINNA.A1.data.clamp_(min=0.)
    PINNA.E2.data.clamp_(min=0.)
    PINNA.A2.data.clamp_(min=0.)

    PINNB.E1.data.fill_(float(PINNA.E1.detach().numpy()))
    PINNB.A1.data.fill_(float(PINNA.A1.detach().numpy()))
    PINNB.E2.data.fill_(float(PINNA.E2.detach().numpy()))
    PINNB.A2.data.fill_(float(PINNA.A2.detach().numpy()))
    PINNB.optimizer.step(PINNB.closure)
    PINNB.E1.data.clamp_(min=0.)
    PINNB.A1.data.clamp_(min=0.)
    PINNB.E2.data.clamp_(min=0.)
    PINNB.A2.data.clamp_(min=0.)

    vec_loss_PINNA.append(PINNA.total_loss.detach().numpy())
    vec_loss_PINNB.append(PINNB.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_cA_data: {PINNA.loss_c_data:.4e} \t loss_cA_ode: {PINNA.loss_c_ode:.4e} \t loss_cB_data: {PINNB.loss_c_data:.4e} \t loss_cB_ode: {PINNB.loss_c_ode:.4e}')

    epoch += 1

print(f'Ea1 = {float(PINNA.E1.detach().numpy()):0.4f}')
print(f'A1 = {float(PINNA.A1.detach().numpy()):0.4f}')
print(f'Ea2 = {float(PINNB.E2.detach().numpy()):0.4f}')
print(f'A2 = {float(PINNB.A2.detach().numpy()):0.4f}')

plt.plot(vec_loss_PINNA, label='PINNA')
plt.plot(vec_loss_PINNB, label='PINNB')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

t = X_train[:,0].detach().numpy()
T = Y_train[:,2].detach().numpy()

cA_true = Y_train[:,0].detach().numpy()
cB_true = Y_train[:,1].detach().numpy()

cA_pred = PINNA.PINN(X_train).detach().numpy()
cB_pred = PINNB.PINN(X_train).detach().numpy()

k1_pred = float(PINNA.A1.detach().numpy()) * np.exp(-float(PINNA.E1.detach().numpy()) / T)
k1_true = prm.A1 * np.exp(-prm.E1 / T)
k2_pred = float(PINNB.A2.detach().numpy()) * np.exp(-float(PINNB.E2.detach().numpy()) / T)
k2_true = prm.A2 * np.exp(-prm.E2 / T)

class parameters2():
    k1 = None
    k2 = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = 0.2
    c1 = -0.01
    c2 = 0.1
    A1 = float(PINNA.A1.detach().numpy())
    E1 = float(PINNA.E1.detach().numpy())
    A2 = float(PINNB.A2.detach().numpy())
    E2 = float(PINNB.E2.detach().numpy())

prm2 = parameters2()
X_test, Y_test = create_data(y0, t1, t2, collocation_points, prm2)

plt.plot(t[idx], cA_true[idx], 'o', label='cA_true')
plt.plot(X_test.flatten(), Y_test[:,0], '--', label='cA_test')
plt.plot(t, cA_pred, label='cA_pred')
plt.plot(t[idx], cB_true[idx], 'o', label='cB_true')
plt.plot(X_test.flatten(), Y_test[:,1], '--', label='cB_test')
plt.plot(t, cB_pred, label='cB_pred')
plt.xlabel('Temps [sec]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(T, k1_pred, label='k1_pred')
plt.plot(T, k1_true, '--', label='k1_true')
plt.plot(T, k2_pred, label='k2_pred')
plt.plot(T, k2_true, '--', label='k2_true')
plt.xlabel('Température [°C]')
plt.ylabel('Constante [1/s]')
plt.legend()
plt.show()
