import numpy as np
import pandas as pd
import os

PATH = os.getcwd()

data = {'t':[], 'Q':[], 'T':[]}

for root, dirs, files in os.walk(PATH):
    
    for file in files:
        
        if file.split('.')[-1] == 'csv':

            with open(file) as f:
                lines = f.readlines()

            count = 0
            for line in lines:
                line_content = line.split(';')
                if line_content[0] == 'hh:mm:ss':
                    start = count
                    break
                count += 1
            start += 1

            t = []
            T_ir = []
            for i in range(start,len(lines)-1):
                Power = float(lines[i].split(';')[5])
                if Power == 0.0 and i != start:
                    break
                else:
                    time = lines[i].split(';')[0].split(':')
                    minutes = float(time[1])
                    seconds = float(time[2])
                    t.append(minutes*60+seconds)
                    T_ir.append(float(lines[i].split(';')[2]))
            
            Q = np.ones(len(t))*float(file.split('-')[1].split('W')[0])
            Q = Q.tolist()
            
            data['t'] = data['t'] + t
            data['Q'] = data['Q'] + Q
            data['T'] = data['T'] + T_ir

df = pd.DataFrame.from_dict(data)
df.to_csv('T_train.csv')