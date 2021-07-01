import os
import logging
import configparser
import pickle
import numpy as np

import src.costs.EM_shape_gradient as EM
from src.costs.distance_shape_gradient import Distance_shape_gradient
from src.costs.perimeter_shape_gradient import Perimeter_shape_gradient
from src.costs.curvature_shape_gradient import Curvature_shape_gradient
from src.surface.surface_Fourier import Surface_Fourier

class Simu_result():
    def __init__(self,path) -> None:
        self.path=path
        # we try to load the different file
        try :
            config = configparser.ConfigParser()
            config.read(path+'/config.ini')
            self.config=config
            with open(path+'/result','rb') as file:
                self.result=pickle.load(file)
            self.x=self.result.x
        except :
            raise Exception('unable to deal with the folder {}, TODO handle this exception'.format(path))
        
        self.Np=int(config['geometry']['Np'])
        self.ntheta_coil   = int(config['geometry']['ntheta_coil'])
        self.nzeta_coil   = int(config['geometry']['nzeta_coil'])
        self.lamb=float(self.config['other']['lamb'])
        self.EM=EM.EM_shape_gradient(config=config)
        (m,n,Rmn,Zmn)=self.EM.S_parametrization
        self.m,self.n=m,n
        self.init_param=np.concatenate((Rmn,Zmn))
        self.dist=Distance_shape_gradient(config=config)
        self.perim=Perimeter_shape_gradient(config=config)
        self.curv=Curvature_shape_gradient(config=config)
        self.successful=True
    def get_data_dic(self):
        dic={}
        dic['Name']=self.path
        dic['lambda']=str(self.lamb)
        Penalization=''
        if self.config['optimization_parameters']['d_min']=='True':
            Penalization+='D'
        if self.config['optimization_parameters']['perim']=='True':
            Penalization+='P'
        if self.config['optimization_parameters']['curvature']=='True':
            Penalization+='C'
        dic['Penalization']=Penalization
        R=self.x[:len(self.m)]
        Z=self.x[len(self.m):]
        paramS=((self.m,self.n,R,Z))
        S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        EM_cost,EM_dic=self.EM.cost(S)
        dic['cost B']='{:.2e}'.format(EM_dic['cost_B'])
        dic['max B']='{:.2e}'.format(EM_dic['err_max_B'])
        dic['cost j']='{:.2e}'.format(EM_dic['cost_J'])
        dic['max j']='{:.2e}'.format(EM_dic['max_j'])
        dic['EM cost']='{:.2e}'.format(EM_cost)
        dist_cost,dist_dic=self.dist.cost(S)
        dic['Distance']='{:.2e}'.format(dist_dic['min_distance'])
        perim_cost,perim_dic=self.perim.cost(S)
        dic['perimeter']='{:.2e}'.format(perim_dic['perimeter'])
        curv_cost,curv_dic=self.curv.cost(S)
        dic['Max curv']='{:.2e}'.format(curv_dic['max_curvature'])
        dic['nit']=str(self.result.nit)
        self.S=S
        return dic
        #'Name', 'lambda', 'Penalization', 'cost B', 'max B', 'cost j', 'max j', 'EM cost', 'Distance', 'perimeter', 'Max curv')
    def get_data_dic_ref(self):
        #We use the ini_param instead of the optimize object
        dic={}
        dic['Name']='ref'
        dic['lambda']=str(self.lamb)
        Penalization='None'
        dic['Penalization']=Penalization
        R=self.init_param[:len(self.m)]
        Z=self.init_param[len(self.m):]
        paramS=((self.m,self.n,R,Z))
        S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        EM_cost,EM_dic=self.EM.cost(S)
        dic['cost B']='{:.2e}'.format(EM_dic['cost_B'])
        dic['max B']='{:.2e}'.format(EM_dic['err_max_B'])
        dic['cost j']='{:.2e}'.format(EM_dic['cost_J'])
        dic['max j']='{:.2e}'.format(EM_dic['max_j'])
        dic['EM cost']='{:.2e}'.format(EM_cost)
        dist_cost,dist_dic=self.dist.cost(S)
        dic['Distance']='{:.2e}'.format(dist_dic['min_distance'])
        perim_cost,perim_dic=self.perim.cost(S)
        dic['perimeter']='{:.2e}'.format(perim_dic['perimeter'])
        curv_cost,curv_dic=self.curv.cost(S)
        dic['Max curv']='{:.2e}'.format(curv_dic['max_curvature'])
        dic['nit']=''
        self.S_ref=S
        return dic
def load_all_simulation(path):
    #load all the simulation folders in the given path
    lst_simu=[]
    with os.scandir(path) as it :
        for entry in it:
            if not entry.name.startswith('.'):
                print(entry.name)
                flag=Simu_result(path+entry.name)
                if flag.successful:
                    lst_simu.append(flag)
    return lst_simu
if __name__=='__main__':
    print(load_all_simulation('tmp/')[0].get_data_dic())