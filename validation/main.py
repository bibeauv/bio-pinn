from numerical import *
import pandas as pd
import matplotlib.pyplot as plt

T_df = pd.read_csv('T_train.csv')
GC_df = pd.read_csv('GC.csv')
GC_df = GC_df.fillna(0.0)

class parameters():
    A1 = 0.013784660957753658
    E1 = 0.0006195656605996192
    A2 = 0.014179264195263386
    E2 = 0.0008525238954462111
    A3 = 0.009907885454595089
    E3 = 0.0004600012907758355
    A4 = 0.011547332629561424
    E4 = 0.0007359213195741177
    A5 = 0.002533452585339546
    E5 = 0.00011781039938796312
    A6 = 0.011532556265592575
    E6 = 0.0010021203197538853
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

data_4W = {}
data_5W = {}
data_6W = {}

data_4W['TG'] = y_num_4W[:,0]
data_5W['TG'] = y_num_5W[:,0]
data_6W['TG'] = y_num_6W[:,0]
data_4W['DG'] = y_num_4W[:,1]
data_5W['DG'] = y_num_5W[:,1]
data_6W['DG'] = y_num_6W[:,1]
data_4W['MG'] = y_num_4W[:,2]
data_5W['MG'] = y_num_5W[:,2]
data_6W['MG'] = y_num_6W[:,2]

pd.DataFrame.from_dict(data_4W).to_csv('data_4W.csv')
pd.DataFrame.from_dict(data_5W).to_csv('data_5W.csv')
pd.DataFrame.from_dict(data_6W).to_csv('data_6W.csv')

plt.plot(t_num_4W, y_num_4W[:,0], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,0], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,0], '--', label='Numerical 6W')
plt.errorbar(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['TG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 4.0]['erreur_TG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 4W')
plt.errorbar(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['TG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 5.0]['erreur_TG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 5W')
plt.errorbar(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['TG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 6.0]['erreur_TG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 6W')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,1], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,1], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,1], '--', label='Numerical 6W')
plt.errorbar(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['DG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 4.0]['erreur_DG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 4W')
plt.errorbar(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['DG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 5.0]['erreur_DG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 5W')
plt.errorbar(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['DG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 6.0]['erreur_DG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 6W')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t_num_4W, y_num_4W[:,2], '--', label='Numerical 4W')
plt.plot(t_num_5W, y_num_5W[:,2], '--', label='Numerical 5W')
plt.plot(t_num_6W, y_num_6W[:,2], '--', label='Numerical 6W')
plt.errorbar(GC_df[GC_df['Power'] == 4.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 4.0]['MG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 4.0]['erreur_MG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 4W')
plt.errorbar(GC_df[GC_df['Power'] == 5.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 5.0]['MG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 5.0]['erreur_MG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 5W')
plt.errorbar(GC_df[GC_df['Power'] == 6.0]['Time'].to_numpy(), GC_df[GC_df['Power'] == 6.0]['MG'].to_numpy(), yerr=GC_df[GC_df['Power'] == 6.0]['erreur_MG'].to_numpy(), fmt='o', ms=4, capsize=4, label='GC 6W')
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
