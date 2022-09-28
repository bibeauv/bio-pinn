import matplotlib.pyplot as plt
import numpy as np
from biopinn import BioPINN as bp
import pandas as pd

# 

variables = ['t']

functionals = ['cB','cTG','cDG','cMG','cG']

parameters = [{'name':'k1', 'initial_value':4.271144},
              {'name':'k2', 'initial_value':0.4526329},
              {'name':'k3', 'initial_value':5.070643},
              {'name':'k4', 'initial_value':1.2533575},
              {'name':'k5', 'initial_value':6.002198},
              {'name':'k6', 'initial_value':0.01145097}]

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

pinn = bp(variables, functionals, parameters, odes)
pinn.set_data('bio.csv', mul=1000)
pinn.set_model(layers=3, neurons=10, optimizer='adam')
pinn.start_training(epochs=10000, batch_size=4001)
print(pinn.get_kinetics())
pinn.generate_loss_graph()
for f in functionals:
    pinn.generate_graph(f, dt_euler=0.01)