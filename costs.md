# Cost related classes

`AbstractCost` is used to define :

* Multiple different costs :
  * `DistanceCost`
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
