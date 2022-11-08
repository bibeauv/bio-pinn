from biopinn import BioPINN as bp
from matplotlib import animation
import matplotlib.pyplot as plt
import os

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
    
# Animation
fig, ax = plt.subplots()

def animate(i):
    if i % 2 == 0:
        pinn = bp(variables, functionals, parameters, odes)
        pinn.set_data('bio.csv', mul=1000)
        pinn.set_model(layers=7, neurons=10)
        pinn.start_training(epochs=i, batch_size=10000, verbose=0)
        
        ax.clear()
        concentrations = pinn.get_concentrations()
        kinetics = pinn.get_kinetics()
        k1 = kinetics['k1']
        k2 = kinetics['k2']
        ax.plot(pinn.X[0], concentrations['cTG'], label=f'k1 = {k1[0]:.2f}, k2 = {k2[0]:.2f}')
        ax.plot(pinn.X[0].flatten()[pinn.ids[1][0].flatten()], pinn.y[1], 'o')
        ax.set_xlim([-0.1,8.1])
        ax.set_ylim([-0.1,0.75])
        ax.set_title(f'Epochs = {i}')
        ax.legend()
    
        del pinn
        
ani = animation.FuncAnimation(fig, animate, frames=10, repeat=False)
writergif = animation.FFMpegWriter(fps=10)
ani.save('animation.gif', writer=writergif)
