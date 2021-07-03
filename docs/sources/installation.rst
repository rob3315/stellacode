Installation
------------------

1. Check that `python 3 <https://www.python.org/downloads/>`_ , `Anaconda <https://www.anaconda.com/products/individual>`_ and `git <https://git-scm.com/downloads>`_ are installed.

2. Clone the git repository :
    ::

        git clone https://plmlab.math.cnrs.fr/rrobin/stellacode.git    #clone the repo
        cd stellacode/ #go to the new folder

#. Create a virtual python environment with conda with the needed packages:
    ::

        # create a new virtual environment named stellacode
        conda create --name stellacode python=3.6 matplotlib scipy dask opt_einsum jupyter
        conda activate stellacode # activate the environment
    
4. (optional) Install `Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_ for 3D visualization
    ::

        pip install vtk
        pip install mayavi
    
5. (optional) Run the tests.
    .. warning::

        It can be memory expensive, around 16-20 GB and takes a few minutes.
    
    ::

        python -m unittest discover -s test

1. (very optional) Install the package to compile the documentation (`available here <https://rrobin.pages.math.cnrs.fr/stellacode/>`_ ).
    ::
    
        conda install -c conda-forge pandoc
        pip install -U sphinx
        pip install sphinx-autoapi nbsphinx nbsphinx-link
  