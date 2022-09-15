import matplotlib.pyplot as plt
import numpy as np
from biopinn import BioPINN as bp

# 

variables = ['t']

functionals = ['cB','cTG','cDG','cMG','cG']

parameters = [{'name':'k1', 'initial_value':0.1},
              {'name':'k2', 'initial_value':0.0},
              {'name':'k3', 'initial_value':1.0},
              {'name':'k4', 'initial_value':0.0},
              {'name':'k5', 'initial_value':0.5},
              {'name':'k6', 'initial_value':0.0}]

odes = [{'sp':'cB',
         'gen':[['k1', 'cTG'],['k3', 'cDG'],['k5', 'cMG']],
         'cons':[['k2', 'cB', 'cDG'],['k4', 'cB', 'cMG'],['k6', 'cB', 'cG']]
         },
        
        {'sp':'cTG',
         'gen':[['k2', 'cB', 'cDG']],
         'cons':[['k1', 'cTG']]
         },
        
        {'sp':'cDG',
         'gen':[['k4', 'cB', 'cMG']],
         'cons':[['k3', 'cDG']]
         },
        
        {'sp':'cMG',
         'gen':[['k6', 'cB', 'cG']],
         'cons':[['k5', 'cMG']]
         },
        
        {'sp':'cG',
         'gen':[['k5', 'cMG']],
         'cons':[['k6', 'cB', 'cG']]
         }]

pinn = bp(variables, functionals, parameters, odes)
pinn.set_data('bio.csv', mul=50)
pinn.set_model(layers=4, neurons=10, optimizer='adam')
pinn.start_training(epochs=1000, batch_size=151)
pinn.print_k()
for fun in functionals:
    pinn.generate_graph(fun=fun)