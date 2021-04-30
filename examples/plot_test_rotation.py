from toroidal_surface import *

def expand_for_plot(X,Y,Z):
    """from a toroidal_surface surface return X,Y,Z
    and add redundancy of first row"""
    shape=(X.shape[0]+1,X.shape[1])
    lst=[]
    for elt in [X,Y,Z]:
        new_elt=np.zeros(shape)
        new_elt[:-1,:]=elt
        new_elt[-1,:]=elt[0,:]
        lst.append(new_elt.copy())
    return lst


def plot(S,lst):
    from mayavi import mlab
    X,Y,Z=expand_for_plot(S.X,S.Y,S.Z)
    s = mlab.mesh(X,Y,Z,representation='mesh',colormap='Wistia')
    X2,Y2,Z2=expand_for_plot(lst[0],lst[1],lst[2])
    s2 = mlab.mesh(X2,Y2,Z2,representation='mesh',colormap='Wistia')
    mlab.show()

if __name__=='__main__':
    Np=3
    S_parametrization=Toroidal_surface.load_file('code/data/li383/cws.txt')
    S=Toroidal_surface(S_parametrization,(128,128),Np)
    pos=np.array([S.X,S.Y,S.Z])# tensor 3 x lu1 x lv1
    rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
    #we rotate the first surface:
    i=1
    X_r,Y_r,Z_r=np.einsum('ij,jkl->ikl',np.linalg.matrix_power(rot,i),pos)# rotation around
    plot(S,[X_r,Y_r,Z_r])