import sciann as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BioPINN():
    
    def __init__(self, variables, functionals, parameters, odes):
        
        # Data
        self.X = []
        self.y = []
        self.ids = []
        
        # Model
        self.m = None
        self.variables = variables
        self.functionals = functionals
        self.parameters = parameters
        self.odes = odes
        self.evaluations = []
    
    def set_data(self, data_file, mul=2):
        """Function to set X and y from the GC data

        Args:
            data_file (string): Name of the file containing the GC results
            mul (int, optional): Multiplicator for collocation points. Defaults to 2.

        Raises:
            Exception: If the data does not fit the functionals of the PINN set by the user
        """
        
        data = pd.read_csv(data_file)
        collocation_points = (len(data)-1)*mul+1
        i = 0
        for col in data.columns:
            if col == 't':
                self.X = [np.linspace(data['t'].iloc[0],
                                      data['t'].iloc[-1],
                                      collocation_points).reshape(-1,1)]
            elif col == self.functionals[i]:
                self.y.append(data[self.functionals[i]].to_numpy())
                i += 1
            else:
                raise Exception("The functionals set for the PINN does not fit the data!")
                
        for i in range(0,len(self.functionals)):
            self.ids += [(np.arange(0,collocation_points,mul).reshape(-1,1),
                          self.y[i].reshape(-1,1))]

    def set_model(self,
              layers=1,
              neurons=1,
              activation='tanh',
              initializer='glorot_uniform',
              optimizer='adam'):
        
        """Function to set the model of the PINN

        Raises:
            Exception: If the first functionnal is not the time of the reaction
        """
        
        list_of_variables = str(self.variables).replace("'","")
        if 't' != self.variables[0]:
            raise Exception("The first variable should always be the time!")
        for v in self.variables:
            exec(f"{v} = sn.Variable()")
        
        list_of_functionals = str(self.functionals).replace("'","")
        for c in self.functionals:
            exec(f"{c} = sn.Functional('{c}', {list_of_variables}, {layers}*[{neurons}], activation='{activation}', kernel_initializer='{initializer}')")
            exec(f"self.evaluations.append({c})")
        list_of_functionals = list_of_functionals.replace(']',',')
           
        for p in self.parameters:
            exec(f"{p['name']} = sn.Parameter({p['initial_value']}, inputs={list_of_variables})")
        
        list_of_odes = []
        for i, ode in enumerate(self.odes):
            exec(f"list_of_odes.append('L{i}')")
            exec_ode = f"L{i} = sn.math.diff({ode['sp']}, t)"
            for j in range(1,len(ode['cons'])):
                if j == 1:
                    exec_ode += f" + {ode['cons'][0]}*{ode['cons'][j]}"
                else:
                    exec_ode += f" * {ode['cons'][j]}"
            for j in range(1,len(ode['gen'])):
                if j == 1:
                    exec_ode += f" - {ode['gen'][0]}*{ode['gen'][j]}"
                else:
                    exec_ode += f" * {ode['gen'][j]}"
            exec(exec_ode)
        list_of_odes = str(list_of_odes).replace("'","")
        list_of_odes = list_of_odes.replace('[',' ')
        
        exec(f"self.m = sn.SciModel({list_of_variables}, {list_of_functionals + list_of_odes}, optimizer='{optimizer}')")
        
    def start_training(self, epochs=100, batch_size=1):
        
        """Function that trains the PINN
        """
        
        x_true = self.X
        y_true = self.ids + len(self.odes)*['zeros']
        self.m.train(x_true,
                     y_true,
                     epochs=epochs,
                     batch_size=batch_size)

    def generate_graph(self):

        """Generate the graph to show results of predictions

        Raises:
            Exception: If the PINN model does not exist, thus no graph can be generated
        """

        if self.m is not None:
            for i, evaluation in enumerate(self.evaluations):
                y_pred = evaluation.eval(self.X)
                plt.plot(self.X[0], y_pred, label=evaluation.name+' pred')
                plt.plot(self.X[0].flatten()[self.ids[i][0].flatten()], self.y[i], 'o', label=evaluation.name+' true')
            plt.legend()
            plt.xlabel('Time [min]')
            plt.ylabel('Concentration [mol/L]')
            plt.show()
        else:
            raise Exception("Can not generate graph, no PINN model exists!")