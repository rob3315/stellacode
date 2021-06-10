Python code for cws shape optimization.
to install:
first install python, conda and git
Clone the repo :
git clone https://plmlab.math.cnrs.fr/rrobin/stellacage_code.git
cd stellacage_code/
Then create a virtual python environnment with the needed package :
conda create --name stellacage_code python=3.6 matplotlib scipy dask opt_einsum jupyter
conda activate stellacage_code
For 3D visualization
pip install vtk
pip install mayavi
