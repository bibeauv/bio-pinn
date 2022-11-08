from biopinn import BioPINN as bp

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

pinn = bp(variables, functionals, parameters, odes)
pinn.set_data('bio.csv', mul=1000)
pinn.set_model(layers=7, neurons=10)
pinn.start_training(epochs=100000, batch_size=10001)
print(pinn.get_kinetics())
pinn.generate_loss_graph()
for f in functionals:
    pinn.generate_graph(f, dt_euler=0.01)