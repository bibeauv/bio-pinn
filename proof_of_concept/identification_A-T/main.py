from data import *
from pinn import *
import matplotlib.pyplot as plt

y0 = np.array([1., 35.])
t1 = 0
t2 = 300
collocation_points = 301

class parameters():
    k = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = 0.2
    c1 = -0.01
    c2 = 0.1
    A = 0.08
    Ea = 100

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
max_epochs = 100000
vec_loss = []
while epoch <= max_epochs:

    PINN.optimizer.step(PINN.closure)

    PINN.Ea.data.clamp_(min=0.)
    PINN.A.data.clamp_(min=0.)

    vec_loss.append(PINN.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e} \t loss_T_data: {PINN.loss_T_data:.4e} \t loss_T_ode: {PINN.loss_T_ode:.4e}')

    if epoch == 50000:
        PINN.optimizer = torch.optim.LBFGS(PINN.params)

    epoch += 1

print(f'Ea = {float(PINN.Ea.detach().numpy())}')
print(f'A = {float(PINN.A.detach().numpy())}')
print(f'e = {float(PINN.e.detach().numpy())}')
print(f'c1 = {float(PINN.c1.detach().numpy())}')
print(f'c2 = {float(PINN.c2.detach().numpy())}')

plt.plot(vec_loss)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

t = X_train[:,0].detach().numpy()

T_true = Y_train[:,1].detach().numpy()
T_pred = PINN.PINN(X_train)[:,1].detach().numpy()

c_true = Y_train[:,0].detach().numpy()
c_pred = PINN.PINN(X_train)[:,0].detach().numpy()

k_pred = float(PINN.A.detach().numpy()) * np.exp(-float(PINN.Ea.detach().numpy()) / T_pred)
k_true = prm.A * np.exp(-prm.Ea / T_true)

class parameters2():
    k = None
    m_Cp = 10
    dHrx = -45000
    Q = 5
    V = 5 / 1000
    e = float(PINN.e.detach().numpy())
    c1 = float(PINN.c1.detach().numpy())
    c2 = float(PINN.c2.detach().numpy())
    A = float(PINN.A.detach().numpy())
    Ea = float(PINN.Ea.detach().numpy())

prm2 = parameters2()
X_test, Y_test = create_data(y0, t1, t2, collocation_points, prm2)

plt.plot(t[idx], c_true[idx], 'o', label='c_true')
plt.plot(t, c_pred, label='c_pred')
plt.plot(X_test.flatten(), Y_test[:,0], '--', label='c_test')
plt.legend()
plt.show()

plt.plot(t[idx], T_true[idx], 'o', label='T_true')
plt.plot(t, T_pred, label='T_pred')
plt.plot(X_test.flatten(), Y_test[:,1], '--', label='T_test')
plt.legend()
plt.show()

plt.plot(T_pred, k_pred, label='k_pred')
plt.plot(T_true, k_true, '--', label='k_true')
plt.legend()
plt.show()
