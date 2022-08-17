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

    # If no bnorm file has been given, generate it from wout using BNORM.
    if not config['other']['path_bnorm']:
        from src.tools.generate_bnorm import generate_bnorm
        generate_bnorm(config)

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

    # Getting bounds and fixed parameters
    file_extension = config['geometry']['path_cws'].split('.')[-1]
    if file_extension == "json":
        opti_parameters, bounds = get_bounds_from_json(
            config['geometry']['path_cws'])
    else:
        opti_parameters, bounds = None, None

    if opti_parameters is None:  # no fixed parameters, no bounds

        opti_parameters = np.full_like(full_grad.init_param, True)

    if bounds is None:  # no boundaries

        opti_method = 'BFGS'

    else:

        opti_method = 'SLSQP'
        from scipy.optimize import Bounds
        bounds = Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=True)

    def f(X, info):
        new_param = np.copy(full_grad.S.param)
        new_param[opti_parameters] = X
        res = full_grad.cost(new_param)
        # display information
        if info['Nfeval'] % freq_save == 0:
            logging.warning(
                'Neval : {0:4d} \n saving the intermediate shape'.format(info['Nfeval']))
            with open('{}/intermediate_param{:}.res'.format(output_folder_name, info['Nfeval']), 'wb') as output_file:
                pickle.dump(full_grad.S.param, output_file)
            with open('{}/intermediate_current{:}.res'.format(output_folder_name, info['Nfeval']), 'wb') as output_file:
                pickle.dump(full_grad.get_j_3D(), output_file)
        info['Nfeval'] += 1
        return res

    def gradf(X, info):
        new_param = np.copy(full_grad.S.param)
        new_param[opti_parameters] = X
        return full_grad.shape_gradient(new_param)[opti_parameters]

    # The optimization
    init_param = full_grad.init_param[opti_parameters]

    optimize_shape = scipy.optimize.minimize(f, init_param, jac=gradf, args=(
        {'Nfeval': 0},), method=opti_method, bounds=bounds, options={'maxiter': max_iter})
    # We save the
    logging.warning('optimization ended, saving file')
    with open('{}/result'.format(output_folder_name), 'wb') as output_file:
        pickle.dump(optimize_shape, output_file)
    with open('{}/current_distribution'.format(output_folder_name), 'wb') as output_file:
        pickle.dump(full_grad.get_j_3D(), output_file)


def get_bounds_from_json(path_cws):
    """
    Only works for cylinders.
    """
    import json
    with open(path_cws, 'r') as f:
        data = json.load(f)

    opti_parameters = []
    bounds = []

    def add_bound(low_bound, high_bound):
        if low_bound is None:
            low_bound = - np.inf
        if high_bound is None:
            high_bound = np.inf
        if low_bound == high_bound:
            opti_parameters.append(False)
        else:
            opti_parameters.append(True)
            bounds.append((low_bound, high_bound))

    if "bounds" in data.keys():

        if data['surface']['parametrization'] == "fourier":

            cos_bounds, sin_bounds = data['bounds'].values()[:2]

            for low_bound, high_bound in cos_bounds:
                add_bound(low_bound, high_bound)

            for low_bound, high_bound in sin_bounds:
                add_bound(low_bound, high_bound)

            for low_bound, high_bound in data['bounds'].values()[2:]:
                add_bound(low_bound, high_bound)

        if data['surface']['parametrization'] == "ell_tri":

            for low_bound, high_bound in data['bounds'].values():
                add_bound(low_bound, high_bound)
        else:

            raise(ValueError, "Unrecognized parametrization.")

        opti_parameters = np.array(opti_parameters)
        bounds = np.array(bounds)

        # no bounds FOR OPTIMIZATION PARAMETERS
        if np.all(bounds[:, 0] == - np.inf) and np.all(bounds[:, 1] == np.inf):
            bounds = None

        return opti_parameters, bounds

    else:  # all parameters are free to move

        return None, None


if __name__ == '__main__':
    # launch the script with the argument of the call python launcher.py path_to_config
    launch(sys.argv[1])
    # launch('config_file/config_full.ini')
