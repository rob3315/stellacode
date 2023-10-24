# Concepts in stellacode

This document tries to define the different classes and concepts used and defined in stellacode.

Stellacode uses two main categories of classes:
* Surface related classes:
  * Surface: a container for the surface position, jacobian etc... Surfaces are used to represent both the coils and the plasmas.
  * CoilSurface: a surface plus a current flowing on the surface.
  * CoilOperator: takes as input current parameters and returns a CoilSurface. Its use is to compute the Biot Savard Operator for solving for the current in one single regression instead of a gradient descent algorithm.
  * Surface factories inheriting from AbstractBaseFactory, whose goal is to fabricate surfaces:
     * Some factories inherit from AbstractSurfaceFactory and fabricate basic surfaces (cylinders etc...), these typically need no inputs.
     * Others take surfaces are inputs and return transformed surfaces (they are in factory_tools):
        * RotatedSurface: transform one surface in a number of duplicated and rotated surfaces
        * ConcatSurfaces: Apply a list of surface factories and concatenates the resulting surfaces along the toroidal dimensions.
        * Sequential: Apply a list of surface factories one after the other on a surface.
     * CoilFactory: takes as input a Surface and return a CoilOperator or a CoilSurface. A CoilFactory needs an instance of an AbstractCurrent to compute the 2D current operator from which the currents are computed.
     * AbstractToroidalCoils: a tentative to define a general API for a global set of coils (Work In Progress). So that operations such as scaling or setting current parameters can be done without adapting to the details of each factory.
* Costs: inherit from AbstractCost and compute costs for optimizations and take as input a CoilSurface or a CoilOperator.
  * Typically an optimization is run on the sum of a list of costs (using the AggregateCost class)
    

An optimization is run using the Optimizer which takes as input a plasma surface, a coil factory and a Cost.
