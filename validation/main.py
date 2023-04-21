from numerical import *
import pandas as pd
import matplotlib.pyplot as plt

T_df = pd.read_csv('T_train.csv')
GC_df = pd.read_csv('GC.csv')

class parameters():
    A1 = 0.0005568796186707914
    E1 = 0.00036747573176398873
    A2 = 0.009908833540976048
    E2 = 0.0015684497775509953
    A3 = 0.006970762740820646
    E3 = 0.0003831649664789438
    A4 = 0.015245305374264717
    E4 = 0.002910907380282879
    A5 = 0.02748340182006359
    E5 = 0.0
    A6 = 0.004419421311467886
    E6 = 0.000663719663862139
    T = None
prm = parameters()

def get_numerical(P):
    prm.T = T_df[T_df['Power'] == P]['Temperature'].to_numpy()
    y0 = np.array([0.596600966,0.038550212,0.000380326,0.0,0.0])
    t1 = T_df[T_df['Power'] == P]['Time'].to_numpy()[0]
    t2 = T_df[T_df['Power'] == P]['Time'].to_numpy()[-1]
    len_t = T_df[T_df['Power'] == P]['Time'].to_numpy().shape[0]
    t = np.linspace(t1, t2, len_t)
    t_num, y_num = euler(y0, t, prm)

    return t_num, y_num

t_num_4W, y_num_4W = get_numerical(4.0)
t_num_5W, y_num_5W = get_numerical(5.0)
t_num_6W, y_num_6W = get_numerical(6.0)

plt.plot(t_num_4W, y_num_4W[:,0], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,0], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,0], '--', label='Numerical 6W')
plt.plot(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['TG'].to_numpy(), 'o', label='GC 4W')
plt.plot(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['TG'].to_numpy(), 'o', label='GC 5W')
plt.plot(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['TG'].to_numpy(), 'o', label='GC 6W')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,1], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,1], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,1], '--', label='Numerical 6W')
plt.plot(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['DG'].to_numpy(), 'o', label='GC 4W')
plt.plot(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['DG'].to_numpy(), 'o', label='GC 5W')
plt.plot(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['DG'].to_numpy(), 'o', label='GC 6W')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,2], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,2], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,2], '--', label='Numerical 6W')
plt.plot(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['MG'].to_numpy(), 'o', label='GC 4W')
plt.plot(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['MG'].to_numpy(), 'o', label='GC 5W')
plt.plot(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['MG'].to_numpy(), 'o', label='GC 6W')
plt.xlabel('Time [sec]')
plt.ylabel('MG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,3], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,3], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,3], '--', label='Numerical 6W')
plt.xlabel('Time [sec]')
plt.ylabel('G Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,4], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,4], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,4], '--', label='Numerical 6W')
plt.xlabel('Time [sec]')
plt.ylabel('ME Concentration [mol/L]')
plt.legend()
plt.show()