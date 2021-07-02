Installation
------------------

1. Check that `python 3 <https://www.python.org/downloads/>`_ , `Anaconda <https://www.anaconda.com/products/individual>`_ and `git <https://git-scm.com/downloads>`_ are installed.

2. Clone the git repository :
    ::

        git clone https://plmlab.math.cnrs.fr/rrobin/stellacage_code.git    #clone the repo
        cd stellacage_code/ #go to the new folder

#. Create a virtual python environment with conda with the needed package:
    ::

        # create a new virtual environment named stellacage_code
        conda create --name stellacage_code python=3.6 matplotlib scipy dask opt_einsum jupyter
        conda activate stellacage_code # activate the environment
    
4. (optional) install `Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_ for 3D visualization
    ::

        pip install vtk
        pip install mayavi
    
5. (optional) run the tests. It can be memory expensive, around 16-20 GB and takes a few minutes.
    ::

        python -m unittest discover -s test

