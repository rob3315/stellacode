from toroidal_surface import *
import pickle
def expand_for_plot(S):
    """from a toroidal_surface surface return X,Y,Z
    and add redundancy of first row"""
    shape=(S.X.shape[0]+1,S.X.shape[1])
    lst=[]
    for elt in [S.X,S.Y,S.Z]:
        new_elt=np.zeros(shape)
        new_elt[:-1,:]=elt
        new_elt[-1,:]=elt[0,:]
        lst.append(new_elt.copy())
    return lst


def plot(lst_S):
    from mayavi import mlab
    lst_s=[]
    for S in lst_S:
        X,Y,Z=expand_for_plot(S)
        lst_s.append(mlab.mesh(X,Y,Z,representation='mesh',colormap='Wistia'))
    mlab.show()

if __name__=='__main__':
    S_parametrization=Toroidal_surface.load_file('code/data/li383/cws.txt')
    S=Toroidal_surface(S_parametrization,(64,64),3)

    with open('code/data/output/output2','rb') as file:
        new_param=pickle.load(file)
    print(new_param.fun)
    (m,n,Rmn,Zmn)=S_parametrization
    R=new_param.x[:len(m)]
    Z=new_param.x[len(m):]
    S2=Toroidal_surface((m,n,R,Z),(64,64),3)
    Sp_parametrization=Toroidal_surface.load_file('code/data/li383/plasma_surf.txt')
    Sp=Toroidal_surface(Sp_parametrization,(64,64),3)
    plot([S2,Sp])