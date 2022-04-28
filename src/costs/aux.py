""" contained non linear functions (and there derivative) to impose constraints
"""
import logging
import numpy as np


def f_non_linear(d_min_hard, d_min_soft, d_min_penalization, x):
    """blow up before d_min_hard and zero after d_min_soft

    :param d_min_hard: cost is inf for x smaller than this value
    :type d_min_hard: float
    :param d_min_soft: cost is 0 for x bigger than this value
    :type d_min_soft: float
    :param d_min_penalization: a multiplicative constant in the cost
    :type d_min_penalization: float
    :param x: the argument
    :type x: float
    :rtype: float
    """
    if x < d_min_hard:
        logging.info('minimal distance overeached')
        return np.inf
        #raise Exception('minimal distance overeached')
    else:
        return d_min_penalization*np.max((d_min_soft-x, 0))**2/(1-np.max((d_min_soft-x, 0))/(d_min_soft-d_min_hard))


def grad_f_non_linear(d_min_hard, d_min_soft, d_min_penalization, x):
    """gradient of :func:`f_non_linear`
    """
    if d_min_hard < x:
        if x >= d_min_soft:
            return 0
        else:
            c = d_min_soft-d_min_hard
            y = d_min_soft-x
            return d_min_penalization*(-c*y*(2*c - y))/((c - y)**2)
    else:
        logging.info('minimal distance overeached in gradient')
        return -np.inf
        #raise Exception('minimal distance overeached in gradient')


def f_e(c0, c1, x):
    """blow up after c0 and zero before c1

    :param c0: cost is 0 for x smaller than this value
    :type c0: float
    :param c1: cost is inf for x bigger than this value
    :type c1: float
    :param x: the argument
    :type x: float
    :rtype: float
    """
    if 0 <= x and x <= c1:
        return np.max((x-c0, 0))**2/(1 - np.max((x-c0, 0))/(c1-c0))
    else:
        logging.info('maximal value overeached')
        return np.inf
        #raise Exception('infinite cost')


def grad_f_e(c0, c1, x):
    """gradient of :func:`f_e`
    """
    if 0 <= x and x <= c1:
        if x < c0:
            return 0
        else:
            c = c1-c0
            y = x-c0
            return (c*y*(2*c - y))/((c - y)**2)
    else:
        logging.info('maximal value overeached in gradient ')
        return -np.inf
        #raise Exception('infinite cost in gradient')
