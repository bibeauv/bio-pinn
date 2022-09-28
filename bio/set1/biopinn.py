from tabnanny import verbose
import sciann as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class BioPINN():
    
    def __init__(self, variables, functionals, parameters, odes):
        
        # Data
        self.X = []
        self.y = []
        self.ids = []
        
        # Model
        self.m = None
        self.history = None
        self.variables = variables
        self.functionals = functionals
        self.parameters = parameters
        self.odes = odes
        self.evaluations_functionals = []
        self.concentrations = {}
        self.evaluations_parameters = []
        self.kinetics = {}
    
    def set_data(self, data_file, norm=False, mul=2):
        """Function to set X and y from the GC data

        Args:
            data_file (string): Name of the file containing the GC results
            norm(bool, optional): Normalize the data if true
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
            if np.all(self.y[i][1:] == np.zeros(len(self.y[i])-1)):
                self.ids += [(np.array([[0]]),
                              self.y[i][0].reshape(-1,1))]
                self.y[i] = np.array([self.y[i][0]])
            else:
                self.ids += [(np.arange(data['t'].iloc[0],collocation_points,mul).reshape(-1,1),
                              self.y[i].reshape(-1,1))]

    def set_ode(self, exec_ode, ode):

        """Add one ODE to the kinematic system

        Args:
            exec_ode (string): Executable of the ODE
            ode (dict): Information regarding the ODE

        Returns:
            exec_ode (string): New executable of the ODE
        """

        for k in range(len(ode['cons'])):
            for j in range(1,len(ode['cons'][k])):
                if j == 1:
                    exec_ode += f" + {ode['cons'][k][0]}*{ode['cons'][k][j]}"
                else:
                    exec_ode += f"*{ode['cons'][k][j]}"
        for k in range(len(ode['gen'])):
            for j in range(1,len(ode['gen'][k])):
                if j == 1:
                    exec_ode += f" - {ode['gen'][k][0]}*{ode['gen'][k][j]}"
                else:
                    exec_ode += f"*{ode['gen'][k][j]}"

        return exec_ode

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
            exec(f"self.evaluations_functionals.append({c})")
        list_of_functionals = list_of_functionals.replace(']',',')
           
        for p in self.parameters:
            exec(f"{p['name']} = sn.Parameter({p['initial_value']}, inputs={list_of_variables}, name='{p['name']}')")
            exec(f"self.evaluations_parameters.append({p['name']})")
        
        list_of_odes = []
        for i, ode in enumerate(self.odes):
            exec(f"list_of_odes.append('L{i}')")
            exec_ode = f"L{i} = sn.math.diff({ode['sp']}, t)"
            exec_ode = self.set_ode(exec_ode, ode)
            exec(exec_ode)
        list_of_odes = str(list_of_odes).replace("'","")
        list_of_odes = list_of_odes.replace('[',' ')
        
        exec(f"self.m = sn.SciModel({list_of_variables}, {list_of_functionals + list_of_odes}, optimizer='{optimizer}')")
        
    def start_training(self, epochs=100, batch_size=1, verbose=1):
        
        """Function that trains the PINN
        """
        
        x_true = self.X
        y_true = self.ids + len(self.odes)*['zeros']
        self.history =  self.m.train(x_true,
                                     y_true,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose)

    def get_concentrations(self):
        
        """Get concentrations of model
        """
        
        if len(self.concentrations) == 0:
            for evaluation in self.evaluations_functionals:
                self.concentrations[evaluation.name] = evaluation.eval(self.X)

        return self.concentrations
    
    def get_kinetics(self):

        """Get kinetics of model
        """
        
        if len(self.kinetics) == 0:
            for evaluation in self.evaluations_parameters:
                self.kinetics[evaluation.name] = evaluation.eval(self.X)

        return self.kinetics

    def odes_explicite_euler(self, dt):
        
        kinetics = self.get_kinetics()
        for k in kinetics.keys():
            exec(f"{k} = {kinetics[k][0]}")

        conc_names = self.get_concentrations().keys()

        y_euler = np.empty((len(conc_names),1))
        for i, y in enumerate(self.y):
            y_euler[i] = y[0]

        t = 0
        while t < self.X[0][-1][0]:
            for i, c in enumerate(conc_names):
                exec(f"{c} = y_euler[i,-1]")
            y = []
            for i, ode in enumerate(self.odes):
                exec_ode = f"y.append(y_euler[i,-1] + dt*(0"
                exec_ode = self.set_ode(exec_ode, ode) + ')*(-1))'
                exec(exec_ode)
            y_euler = np.append(y_euler, np.array(y).reshape(-1,1), axis=1)
            t += dt

        return y_euler

    def generate_graph(self, functionals_to_evaluate, dt_euler=0):

        """Generate the graph to show results of predictions

        fun:
            functionals_to_evaluate (list of string): Name of functionals to evaluate
            dt_euler (float): Step time if Euler is used for comparison. Step time to 0 by default.
        
        Raises:
            Exception: If the PINN model does not exist, thus no graph can be generated
        """

        if dt_euler > 0:
            y_euler = self.odes_explicite_euler(dt_euler)

        concentrations = self.get_concentrations()
        for i, concentration in enumerate(concentrations.keys()):
            if concentration in functionals_to_evaluate:
                y_pred = concentrations[concentration]
                plt.plot(self.X[0], y_pred, label=concentration+' pred')
                plt.plot(self.X[0].flatten()[self.ids[i][0].flatten()], self.y[i], 'o', label=concentration+' true')
                if dt_euler > 0:
                    plt.plot(np.linspace(self.X[0][0][0], self.X[0][-1][0], np.shape(y_euler)[1]), y_euler[i], '--', label=concentration+' euler')
        plt.legend()
        plt.xlabel('Time [min]')
        plt.ylabel('Concentration [mol/L]')
        plt.show()
    
    def generate_loss_graph(self):

        for l in self.history.history.keys():
            if 'loss' in l:
                plt.plot(self.history.history[l], label=l)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()