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
pinn.generate_loss_graph()
print(pinn.get_kinetics())
for fun in functionals:
    pinn.generate_graph(functionals_to_evaluate=fun, dt_euler=0.001)
