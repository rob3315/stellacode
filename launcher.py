import os
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
import numpy as np
import pickle, logging,time,sys
from src.costs.full_shape_gradient import Full_shape_gradient

path_config=sys.argv[1]
freq_save=100
print('path_config : {}'.format(path_config))
full_grad=Full_shape_gradient(path_config_file=path_config)
output_folder_name='tmp/'+full_grad.config['other']['path_output']
os.mkdir(output_folder_name)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
filehandler = logging.FileHandler('{}/log.txt'.format(output_folder_name), 'a')
formatter = logging.Formatter('%(levelname)s::%(message)s')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

def f(X, info):
    res = full_grad.cost(X)
    # display information
    if info['Nfeval']%freq_save == 0:
        print('Neval : {0:4d} \n saving the intermediate shape'.format(info['Nfeval']))
        with open('{}/intermediate{:}.res'.format(output_folder_name,info['Nfeval']),'wb') as output_file:
            pickle.dump(res,output_file)
    info['Nfeval'] += 1
    return res
def gradf(X, info):
    return full_grad.shape_grad(X)

#We write the config before starting
with open('{}/config.ini'.format(output_folder_name), 'w') as tmp_file:
    full_grad.config.write(tmp_file)
import scipy.optimize
optimize_shape=scipy.optimize.minimize(f, full_grad.init_param, jac=gradf,args=({'Nfeval':0},),options={'maxiter':2000,'return_all':True})
with open('{}/result'.format(output_folder_name),'wb') as output_file:
    pickle.dump(optimize_shape,output_file)