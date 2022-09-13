import sys
import os
from biopinn import BioPINN as bp

# 

variables = ['t']

functionals = ['cA', 'cB', 'cC']

parameters = [{'name':'k1', 'initial_value':1},
              {'name':'k2', 'initial_value':1}]

odes = [{'sp':'cA',
         'gen':[0, 0],
         'cons':['k1', 'cA']},
        
        {'sp':'cB',
         'gen':['k1','cA'],
         'cons':['k2', 'cB']},
        
        {'sp':'cC',
         'gen':['k2','cB'],
         'cons':[0, 0]}]

pinn = bp(variables, functionals, parameters, odes)
pinn.set_data('artificial_data.csv')
pinn.set_model(layers=4, neurons=10)
pinn.start_training(epochs=1000, batch_size=21)
pinn.generate_graph()