import matplotlib.pyplot as plt
import pandas as pd
from source import *

# Data
data = {'E0B':['tests/2022-04-27 17_57_30_canola.txt',      0,      100],
        'E1B':['prise2/2022-10-05 16_13_33_E1minB.txt',     1,      112.8],
        'E2B':['prise2/2022-07-20 11_34_08_E2B.txt',        2,      96.6],
        'E3B':['prise2/2022-07-20 12_09_37_E3B.txt',        4,      99.7],
        'E4B':['prise2/2022-07-20 13_08_15_E4B.txt',        6,      108.8],
        'E5B':['prise2/2022-07-27 12_43_51_E5B.txt',        8,      102.4],
        'E6B':['prise2/2022-07-27 14_40_39_E6B.txt',        10 ,    100]}

# Info of glycerides
info_glycerides = {'mono':[[14.20,14.40],[15.55,16.40]],
                   'di':[[19.9,20.70]],
                   'tri':[[21.3,21.9],[22,22.2],[22.3,22.7]],
                   'istd':[[18.7,19.1]]}

info_glycerol = {'glycerol':[[4.5,4.75]],
                 'istd':[[3.6,3.8]]}

info_biodiesel = {'c16':[[8.7,9],[9,9.3]],
                  'c18.2':[[9.9,11.2]],
                  'c18.3':[[11.5,11.8],[11.8,12]],
                  'istd':[[7.4,8.3]]}

# massentration over time
# ---------------------------------------------------------------
CIS = 8     # mg/mL
VIS = 0.1   # mL

info = info_glycerides

group = ['tri']
# ---------------------------------------------------------------

time = []
mass = []
for exp in data.keys():
    file = data[exp][0]
    t = data[exp][1]
    m = data[exp][2]
    df = read_raw_data(file)
    area = integrate_peaks(df, info)
    
    time.append(t)
    
    group_area = 0
    for g in group:
        group_area += area[g]
    istd_area = area['istd']
    mass.append(group_area / area['istd'] * CIS * VIS)
    
df = pd.DataFrame.from_dict({'-'.join(group):mass})
df.to_csv('-'.join(group)+'.csv',index=False)

plt.plot(time,np.array(mass),'-o',label='-'.join(group))
plt.xlabel('Time')
plt.ylabel('Masse (mg)')
plt.legend()
plt.show()