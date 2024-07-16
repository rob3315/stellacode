# Surface related classes

## Surface objects

`Surface` is a container for :

* A 2D poloidal/toroidal grid of points (u,v) ;
* The surface position in cartesian coordinates (x,y,z) at each (u,v) grid point ;
* The surface jacobian and hessian in cartesian coordinates ;
* The surface normal vector pointing inwards (a priori not unitary) ;
* The meshsize of the grid.

Among the associated methods, one can :

* Measure the distance to another surface using its cartesian coordinates ;
* Integrate a field on the surface ;
* Plot the surface, and optionnally a scalar or vectorial field, in 3D (uses `plotly`) ;
* Plot a 2D field on the (u,v) grid.

### Plasma surfaces

Plasma surfaces are instances of the class `FourierSurface(Surface)`.

### Coil winding surfaces

Coil winding surfaces are instances of the class `CoilSurface(Surface)`
 Surfaces are used to represent both the CWS (when name is `S`) and the plasma (when name is `Sp`). Several classes inherit from it :
  * `CoilSurface` : a surface plus a current flowing on the surface ;
  * `CoilOperator`: takes as input current parameters and returns a `CoilSurface`. Its role is to compute the Biot et Savart Operator to find the current in one single regression instead of doing a gradient descent algorithm ;
  * `FourierSurface` ;

## Surface factories

* `AbstractBaseFactory` fabricates surfaces:

### Basic surfaces

* `AbstractSurfaceFactory` is meant for basic surfaces (cylinders etc...), which typically need no inputs :
  * `VerticalCylinder` ;
  * `ToroidalSurface` ;
  * `CylindricalSurface` ;
  * `FourierSurfaceFactory` ;

### Transformations

* Others take surfaces as inputs and return transformed surfaces (c.f. `factory_tools`):
  * `RotatedSurface`: transform one surface in a number of duplicated and rotated surfaces ;
  * `ConcatSurfaces`: Apply a list of surface factories and concatenates the resulting surfaces along the toroidal dimensions ;
  * `Sequential`: Apply a list of surface factories one after the other on a surface ;

### Coils

* `CoilFactory` takes as input a surface and returns a `CoilOperator` or a `CoilSurface`. A `CoilFactory` needs an instance of an `AbstractCurrent` to compute the 2D current operator from which the currents are computed ;
* `GroovedCoilFactory` ;
* `AbstractToroidalCoils` is a tentative to define a general API for a global set of coils (**WIP**). So that operations such as scaling or setting current parameters can be done without adapting to the details of each factory.