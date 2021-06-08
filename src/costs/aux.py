import logging
import numpy as np

def f_non_linear(d_min_hard,d_min_soft,d_min_penalization,x):
    if x<d_min_hard:
        logging.warning('minimal distance overeached')
        return np.inf
        #raise Exception('minimal distance overeached')
    else :
        return d_min_penalization*np.max((d_min_soft-x,0))**2/(1-np.max((d_min_soft-x,0))/(d_min_soft-d_min_hard))
def grad_f_non_linear(d_min_hard,d_min_soft,d_min_penalization,x):
    if d_min_hard <x :
        if x>=d_min_soft:
            return 0
        else:
            c=d_min_soft-d_min_hard
            y=d_min_soft-x
            return d_min_penalization*(-c*y*(2*c - y))/((c - y)**2)
    else:
        logging.warning('minimal distance overeached in gradient')
        return -np.inf
        #raise Exception('minimal distance overeached in gradient')
def f_e(c0,c1,x):
    if 0 <=x and x <=c1:
        return np.max((x-c0,0))**2/(1- np.max((x-c0,0))/(c1-c0))
    else:
        logging.warning('maximal value overeached')
        return np.inf
        #raise Exception('infinite cost')

def grad_f_e(c0,c1,x):
    if 0 <=x and x <=c1:
        if x<c0:
            return 0
        else:
            c=c1-c0
            y=x-c0
            return (c*y*(2*c - y))/((c - y)**2)
    else:
        logging.warning('maximal value overeached in gradient ')
        return -np.inf
        #raise Exception('infinite cost in gradient')