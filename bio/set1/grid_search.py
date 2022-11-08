import matplotlib.pyplot as plt
import numpy as np
from biopinn import BioPINN as bp
import pandas as pd

# 

variables = ['t']

functionals = ['cB','cTG','cDG','cMG','cG']

parameters = [{'name':'k1', 'initial_value':1.0},
              {'name':'k2', 'initial_value':1.0},
              {'name':'k3', 'initial_value':1.0},
              {'name':'k4', 'initial_value':1.0},
              {'name':'k5', 'initial_value':1.0},
              {'name':'k6', 'initial_value':1.0}]

odes = [{'sp':'cB',
         'gen':[['k1', 'cTG'],['k3', 'cDG'],['k5', 'cMG']],
         'cons':[['k2', 'cDG', 'cB'],['k4', 'cMG', 'cB'],['k6', 'cG', 'cB']]
         },
        
        {'sp':'cTG',
         'gen':[['k2', 'cDG', 'cB']],
         'cons':[['k1', 'cTG']]
         },
        
        {'sp':'cDG',
         'gen':[['k1', 'cTG'],['k4', 'cMG', 'cB']],
         'cons':[['k3', 'cDG'],['k2', 'cDG', 'cB']]
         },
        
        {'sp':'cMG',
         'gen':[['k3', 'cDG'],['k6', 'cG', 'cB']],
         'cons':[['k5', 'cMG'],['k4', 'cMG', 'cB']]
         },
        
        {'sp':'cG',
         'gen':[['k5', 'cMG']],
         'cons':[['k6', 'cG', 'cB']]
         }]

df = {'mul':[],
      'optimizer':[],
      'epochs':[],
      'layers':[],
      'neurons':[],
      'loss':[]}

for mul in [1000]:
    for optimizer in ['adam']:
        for epochs in [1000]:
            for layers in  [5, 6, 7]:
                for neurons in [5, 10, 20, 50]:
                    batch_size = 5*mul+1
                    pinn = bp(variables, functionals, parameters, odes)
                    pinn.set_data('bio.csv', mul=mul)
                    pinn.set_model(layers=layers, neurons=neurons, optimizer=optimizer)
                    pinn.start_training(epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    df['mul'].append(mul)
                    df['optimizer'].append(optimizer)
                    df['epochs'].append(epochs)
                    df['layers'].append(layers)
                    df['neurons'].append(neurons)
                    df['loss'].append(pinn.history.history['loss'][-1])
                    
                    del pinn.m
                    del pinn.history
                    del pinn

df = pd.DataFrame.from_dict(df)
df.to_csv('grid_search.csv')