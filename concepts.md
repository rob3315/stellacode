# Concepts in stellacode

This document tries to define the different classes and concepts used and defined in stellacode.

## Surface related classes

* `Surface` is a container for the surface position, jacobian etc... Surfaces are used to represent both the CWS (when name is `S`) and the plasma (when name is `Sp`). Several classes inherit from it :
  * `CoilSurface` : a surface plus a current flowing on the surface ;
  * `CoilOperator`: takes as input current parameters and returns a `CoilSurface`. Its role is to compute the Biot et Savart Operator to find the current in one single regression instead of doing a gradient descent algorithm ;
  * `FourierSurface` ;

* `AbstractBaseFactory` fabricates surfaces:
    * `AbstractSurfaceFactory` is meant for basic surfaces (cylinders etc...), which typically need no inputs :
      * `VerticalCylinder` ;
      * `ToroidalSurface` ;
      * `CylindricalSurface` ;
      * `FourierSurfaceFactory` ;
    * Others take surfaces as inputs and return transformed surfaces (c.f. `factory_tools`):
      * `RotatedSurface`: transform one surface in a number of duplicated and rotated surfaces ;
      * `ConcatSurfaces`: Apply a list of surface factories and concatenates the resulting surfaces along the toroidal dimensions ;
      * `Sequential`: Apply a list of surface factories one after the other on a surface ;
    * `CoilFactory` takes as input a surface and returns a `CoilOperator` or a `CoilSurface`. A `CoilFactory` needs an instance of an `AbstractCurrent` to compute the 2D current operator from which the currents are computed ;
    * `GroovedCoilFactory` ;
    * `AbstractToroidalCoils` is a tentative to define a general API for a global set of coils (**WIP**). So that operations such as scaling or setting current parameters can be done without adapting to the details of each factory.

## Cost related classes

`AbstractCost` is used to define :
* Multiple different costs :
  * `DistanceCose`
  * `MSEBField`
  * `EMCost`
  * `LaplaceForceCost`
  * `CurvatureCost`
  * `NegTorCurvatureCost`
  * `AreaCost`
  * `CurrentCtrCost`
  * `PoloidalCurrentCost`
  * `CriticalCurrentCtr` 
* `AggregateCost` to sum a list of costs
* `Results` to transmit information from costs to costs

An instance of the `Constraint` class is associated to each cost.

## Optimization class

An optimization is run using an instance of the `Optimizer` class.
It can take as input :
* A plasma surface (methods `from_config_file` and `from_config`) ;
* A coil factory and an `AggregateCost` (method `from_cost`).
One of its attribute is `loss`, the sum of all the costs listed by the `AggregateCost`.