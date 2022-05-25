import configparser
import sys
import time
import logging
import pickle
from src.costs.full_shape_gradient import Full_shape_gradient
import scipy.optimize
import numpy as np
import os
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6


def launch(path_config_file=None, config=None):
    if config is None:
        print('path_config : {}'.format(path_config_file))
        config = configparser.ConfigParser()
        config.read(path_config_file)
    # We create the temporary folder
    output_folder_name = config['other']['path_output']
    os.mkdir(output_folder_name)
    # We save the config file there
    with open('{}/config.ini'.format(output_folder_name), 'w') as tmp_file:
        config.write(tmp_file)
    # log file
    print('initialization of the logger')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(
        '{}/log.txt'.format(output_folder_name), 'a')
    formatter = logging.Formatter('%(levelname)s::%(message)s')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    full_grad = Full_shape_gradient(config=config)
    # optimizer options
    freq_save = int(config['optimization_parameters']['freq_save'])
    max_iter = int(config['optimization_parameters']['max_iter'])

    def f(X, info):
        res = full_grad.cost(X)
        # display information
        if info['Nfeval'] % freq_save == 0:
            logging.warning(
                'Neval : {0:4d} \n saving the intermediate shape'.format(info['Nfeval']))
            with open('{}/intermediate{:}.res'.format(output_folder_name, info['Nfeval']), 'wb') as output_file:
                pickle.dump(res, output_file)
        info['Nfeval'] += 1
        return res

    def gradf(X, info):
        return full_grad.shape_gradient(X)
    # The optimization
    optimize_shape = scipy.optimize.minimize(f, full_grad.init_param, jac=gradf, args=(
        {'Nfeval': 0},), options={'maxiter': max_iter, 'return_all': True})
    # We save the
    logging.warning('optimization ended, saving file')
    with open('{}/result'.format(output_folder_name), 'wb') as output_file:
        pickle.dump(optimize_shape, output_file)
    with open('{}/current_distribution'.format(output_folder_name), 'wb') as output_file:
        pickle.dump(full_grad.get_j_3D(), output_file)


if __name__ == '__main__':
    # launch the script with the argument of the call python launcher.py path_to_config
    launch(sys.argv[1])
    # launch('config_file/config_full.ini')
