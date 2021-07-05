Contribute
==============
**Stellacode** is an open source software and we welcome any contribution.
Feel free to conctact `me <https://rrobin.pages.math.cnrs.fr/#contact>`_ directly or to contribute directly to the `gitlab <https://plmlab.math.cnrs.fr/rrobin/stellacode>`_ .

the ticket list is `here <https://plmlab.math.cnrs.fr/rrobin/stellacode/-/issues>`_ 

Code structure
--------------------
The code structure has been thought to be (relatively) easily extended to other costs and surfaces parametrizations.

::

    src
    ├── costs
    │   ├── abstract_shape_gradient.py
    │   ├── aux.py
    │   ├── curvature_shape_gradient.py
    │   ├── distance_shape_gradient.py
    │   ├── EM_cost.py
    │   ├── EM_shape_gradient.py
    │   ├── full_shape_gradient.py
    │   ├── perimeter_shape_gradient.py
    ├── graphic_interface
    │   └── simu_result.py
    ├── surface
    │   ├── abstract_surface.py
    │   └── surface_Fourier.py
    └── tools
        ├── __init__.py
        └── bnorm.py


Adding a new cost
------------------
