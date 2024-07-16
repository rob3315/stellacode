# Concepts in stellacode

This document tries to define the different classes and concepts used and defined in stellacode.

## Surface related classes

* `Surface` is a container for the surface position, jacobian etc... Surfaces are used to represent both the CWS (when name is `S`) and the plasma (when name is `Sp`). Several classes inherit from it.
* `AbstractBaseFactory` creates surfaces:
  * basic surfaces (cylinders etc...), which typically need no inputs (c.f. `abstract_surface.py`);
  * transformed surfaces, with a surface as input and a transformed surface as output (c.f. `factory_tools.py`) ;
  * coils (c.f. `coil_surface.py`).

For detailed information, see [here](surfaces.md).

## Cost related classes

* `AbstractCost` is used to define several costs for the optimization, each one associated to a `Constraint` class ;
* `Results` is used to transmit the updated current parameters from a cost to another during the optimization of a set of costs.

For detailed information, see [here](costs.md).

## The optimization class

An optimization is run using an instance of the `Optimizer` class.

It can take as input :

* A config file with plasma, CWS and optimization parameters (methods `from_config_file` and `from_config`) ;
* A cost and a coil factory (method `from_cost`).

The optimizer's main attributes are :

* `cost`, an `AggregateCost` instance that lists the costs to be optimized ;
* `coil_factory`, for the optimization of the CWS ;
* `loss`, a function that sums all the costs listed by the `AggregateCost` and updates the CWS and current if applicable ;
* `init_param`, the initial parameters before optimization ;
* `method`, the type of solver to use (see options from [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)).

When running the method `optimize` (no arguments), a gradient descent is called to minimize `loss` for some given initial parameters

The outputs are :

* A total cost value ;
* A dictionnary with the costs metrics (subcosts) ;
* An instance of the class `Results` with the optimized current ;
* A dictionnary with the values of the optimized parameters.

