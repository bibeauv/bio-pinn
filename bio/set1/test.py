import matplotlib.pyplot as plt
import numpy as np
from biopinn import BioPINN as bp

# 

variables = ['t']

functionals = ['cG']

parameters = [{'name':'k1', 'initial_value':0.9}]

odes = [{'sp':'cG',
         'gen':[[0, 0]],
         'cons':[['k1', 'cG']]}]

pinn = bp(variables, functionals, parameters, odes)
pinn.set_data('test.csv', mul=10)
pinn.set_model(layers=4, neurons=10, optimizer='adam')
pinn.start_training(epochs=1000, batch_size=51)
pinn.print_k()
pinn.generate_graph(fun=functionals)