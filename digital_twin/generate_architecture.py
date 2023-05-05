import jinja2
import os
from pyDOE import *
import shutil
import numpy as np

PATH = os.getcwd()
TEMPLATE_PATH = PATH + '/template'
MAIN_FILE = 'main.py'
CASE_PREFIX = 'pinn_'

number_of_cases = 10

k = lhs(2, number_of_cases)

n_min = 10
n_max = 100

l_min = 2
l_max = 4

neurons = np.round(k[:,0]*(n_max - n_min) + n_min, -1).astype(int)
layers = np.round(k[:,1]*(l_max - l_min) + l_min).astype(int)

templateLoader = jinja2.FileSystemLoader(searchpath=TEMPLATE_PATH)
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(MAIN_FILE)

for case in range(number_of_cases):
    parameters = template.render(neurons=neurons[case],
                                 layers=layers[case])
    
    case_folder_name = f'{CASE_PREFIX}{case}'
    case_path = f'{PATH}/{case_folder_name}'
    
    shutil.copytree(TEMPLATE_PATH, case_path)
    
    with open(f'{case_path}/{MAIN_FILE}', 'w') as f:
        f.write(parameters)