import jinja2
import os
from pyDOE import *
import shutil

PATH = os.getcwd()
TEMPLATE_PATH = PATH + '\\template'
MAIN_FILE = 'main.py'
CASE_PREFIX = 'pinn_'

number_of_cases = 2

k = lhs(17, number_of_cases)

E_min = 10
E_max = 500

A_min = 0
A_max = 10

e_min = 0
e_max = 1

c1_min = -1
c1_max = 0

c2_min = 0
c2_max = 1

n_min = 10
n_max = 100

reg_min = 1
reg_max = 1000

E1 = k[:,0]*(E_max - E_min) + E_min
E2 = k[:,1]*(E_max - E_min) + E_min
E3 = k[:,2]*(E_max - E_min) + E_min
E4 = k[:,3]*(E_max - E_min) + E_min
E5 = k[:,4]*(E_max - E_min) + E_min
E6 = k[:,5]*(E_max - E_min) + E_min

A1 = k[:,6]*(A_max - A_min) + A_min
A2 = k[:,7]*(A_max - A_min) + A_min
A3 = k[:,8]*(A_max - A_min) + A_min
A4 = k[:,9]*(A_max - A_min) + A_min
A5 = k[:,10]*(A_max - A_min) + A_min
A6 = k[:,11]*(A_max - A_min) + A_min

e = k[:,12]*(e_max - e_min) + e_min
c1 = k[:,13]*(c1_max - c1_min) + c1_min
c2 = k[:,14]*(c2_max - c2_min) + c2_min

neurons = k[:,15]*(n_max - n_min) + n_min
neurons = np.round(neurons, -1).astype(int)

reg = k[:,16]*(reg_max - reg_min) + reg_min
reg = np.round(reg)

templateLoader = jinja2.FileSystemLoader(searchpath=TEMPLATE_PATH)
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(MAIN_FILE)

for case in range(number_of_cases):
    parameters = template.render(E=[E1[case], E2[case], E3[case], E4[case], E5[case], E6[case]],
                                 A=[A1[case], A2[case], A3[case], A4[case], A5[case], A6[case]],
                                 e=e[case],
                                 c1=c1[case],
                                 c2=c2[case],
                                 neurons=neurons[case],
                                 regularization=reg[case])
    
    case_folder_name = f'{CASE_PREFIX}{case}'
    case_path = f'{PATH}\\{case_folder_name}'
    
    shutil.copytree(TEMPLATE_PATH, case_path)
    
    with open(f'{case_path}\\{MAIN_FILE}', 'w') as f:
        f.write(parameters)