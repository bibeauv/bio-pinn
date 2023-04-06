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

PINN = Curiosity(X_train, Y_train, idx, f_hat, device, prm)

# Make all outputs positive
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 210000
vec_loss = []
check = False
while epoch <= max_epochs:

    PINN.optimizer.step(PINN.closure)

    PINN.E1.data.clamp_(min=0.)
    PINN.A1.data.clamp_(min=0.)
    PINN.E2.data.clamp_(min=0.)
    PINN.A2.data.clamp_(min=0.)

    vec_loss.append(PINN.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e}')

    epoch += 1

print(PINN.E1)
print(PINN.A1)
print(PINN.E2)
print(PINN.A2)

plt.plot(vec_loss)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

t = X_train[:,0].detach().numpy()
T = Y_train[:,2].detach().numpy()

cA_true = Y_train[:,0].detach().numpy()
cB_true = Y_train[:,1].detach().numpy()

cA_pred = PINN.PINN(X_train)[:,0].detach().numpy()
cB_pred = PINN.PINN(X_train)[:,1].detach().numpy()

k1_pred = float(PINN.A1.detach().numpy()) * np.exp(-float(PINN.E1.detach().numpy()) / T)
k1_true = prm.A1 * np.exp(-prm.E1 / T)
k2_pred = float(PINN.A2.detach().numpy()) * np.exp(-float(PINN.E2.detach().numpy()) / T)
k2_true = prm.A2 * np.exp(-prm.E2 / T)

plt.plot(t[idx], cA_true[idx], 'o', label='cA_true')
plt.plot(t, cA_pred, label='cA_pred')
plt.plot(t[idx], cB_true[idx], 'o', label='cB_true')
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
