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

k = lhs(3, number_of_cases)

e_min = 0.5
e_max = 0.5

c1_min = 0.1
c1_max = 0.1

c2_min = 0.1
c2_max = 0.1

e = k[:,0]*(e_max - e_min) + e_min
c1 = k[:,1]*(c1_max - c1_min) + c1_min
c2 = k[:,2]*(c2_max - c2_min) + c2_min

templateLoader = jinja2.FileSystemLoader(searchpath=TEMPLATE_PATH)
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(MAIN_FILE)

for case in range(number_of_cases):
    parameters = template.render(e=e[case],
                                 c1=c1[case],
                                 c2=c2[case])
    
    case_folder_name = f'{CASE_PREFIX}{case}'
    case_path = f'{PATH}/{case_folder_name}'
    
    shutil.copytree(TEMPLATE_PATH, case_path)
    
    with open(f'{case_path}/{MAIN_FILE}', 'w') as f:
        f.write(parameters)