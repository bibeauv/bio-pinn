import numpy as np
import matplotlib.pyplot as plt

def ode(df, y0, k):

    df[0] = k[0]*y0[1] + k[1]*y0[2] + k[2]*y0[3]
    df[1] = -k[0]*y0[1]
    df[2] = -k[1]*y0[2] + k[0]*y0[1]
    df[3] = -k[2]*y0[3] + k[1]*y0[2]
    df[4] = k[2]*y0[3]

    return df

def euler_explicite(y0, dt, tf, k):

    df = np.empty(5)
    y0 = np.array(y0)
    k = np.array(k)

    t = np.array([0])
    y = np.empty((1,len(y0)))
    y[0] = y0
    while t[-1] < tf:
        df = ode(df, y0, k)
        yt = y0 + dt * df
        y = np.append(y,[yt],axis=0)
        y0 = np.copy(yt)
        t = np.append(t,t[-1]+dt)

    return y, t

y, t = euler_explicite([0,0.5,0.06,9e-5,0], 0.01, 10, [1,1,1])

fig, axs = plt.subplots(2,3)

axs[0,0].plot(t,y[:,0])
axs[0,0].set_title('cB')
axs[0,1].plot(t,y[:,1])
axs[0,1].set_title('cTG')
axs[0,2].plot(t,y[:,2])
axs[0,2].set_title('cDG')
axs[1,0].plot(t,y[:,3])
axs[1,0].set_title('cMG')
axs[1,1].plot(t,y[:,4])
axs[1,1].set_title('cG')
axs[-1,-1].axis('off')

plt.show()