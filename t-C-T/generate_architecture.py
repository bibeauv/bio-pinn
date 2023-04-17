import jinja2
import os
from pyDOE import *
import shutil
import numpy as np

PATH = os.getcwd()
TEMPLATE_PATH = PATH + '/template'
MAIN_FILE = 'main.py'
CASE_PREFIX = 'pinn_'

number_of_cases = 2

k = lhs(12, number_of_cases)

E_min = 1
E_max = 1

A_min = 1/240
A_max = 1/240

E1 = k[:,0]*(E_max - E_min) + E_min
E2 = k[:,1]*(E_max - E_min) + E_min
E3 = k[:,2]*(E_max - E_min) + E_min
E4 = k[:,3]*(E_max - E_min) + E_min
E5 = k[:,4]*(E_max - E_min) + E_min
E6 = k[:,5]*(E_max - E_min) + E_min

A1 = k[:,0]*(A_max - A_min) + A_min
A2 = k[:,1]*(A_max - A_min) + A_min
A3 = k[:,2]*(A_max - A_min) + A_min
A4 = k[:,3]*(A_max - A_min) + A_min
A5 = k[:,4]*(A_max - A_min) + A_min
A6 = k[:,5]*(A_max - A_min) + A_min

templateLoader = jinja2.FileSystemLoader(searchpath=TEMPLATE_PATH)
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(MAIN_FILE)

for case in range(number_of_cases):
    parameters = template.render(E=[E1[case], E2[case], E3[case], E4[case], E5[case], E6[case]],
                                 A=[A1[case], A2[case], A3[case], A4[case], A5[case], A6[case]])
    
    case_folder_name = f'{CASE_PREFIX}{case}'
    case_path = f'{PATH}/{case_folder_name}'
    
    shutil.copytree(TEMPLATE_PATH, case_path)
    
    with open(f'{case_path}/{MAIN_FILE}', 'w') as f:
        f.write(parameters)