# SLOHacks2019-Moses

The Moses function, named after Moses Kim, is used as a Google Cloud Function.
The function takes in two sets of data points - the `reference` and the `recent`.
The `recent` data set is processed using a polynomial fit function from numpy for
each axis (x, y, z). Since measurements are recorded at constant intervals, they
are normalized into 3 position functions, one per axis.

The `recent` data set is compared to the fitted position functions. The difference
between the two is recorded. These differences are then reduced into a single value
by taking the standard deviation. This is the error value on a given axis which is
provided back in the JSON response.
