NNS-RPY2-ScikitLearn
============
NNS RPY2  is a python package (pipy) that create a Scikit-Learn interface to 
NNS library (developed in R) with RPY2 package.

This project is a beta version, classifier and regressor classes are exported 
to user.

A second step is export all NNS functions to python world.

A last step is port the NNS R code to python, checking if Cython could 
help with performace and reduce the memory when copying data from python to R

More about NNS: https://cran.r-project.org/package=NNS

Installation
------------
Execute the standard pip install:

```pip install NNS-RPY2-ScikitLearn```


Dependencies
------------

NNS-RPY2-ScikitLearn requires:

- Python (>= 3.6)
- RPY2 (>= 3.2.4)
- Scikit-learn (>= 0.22.1)
- NumPy (>= 1.13.3)


Example
-------

Python code:

```{python}
from nns_rpy2 import NNSRPY2Regressor
model = NNSRPY2Regressor()
x = np.array([1, 2, 3, 4])
x_new = x + 1
y = x ** 3
model.fit(x, y)
print(model.predict(x_new))
```


Contribuiton
------------

We welcome new contributors of all experience levels.
Open a issue, send a pullrequest or start wiki page.


Autors
------
Fred Viole - https://www.linkedin.com/in/fred-viole-9481278/

Roberto Spadim - https://www.linkedin.com/in/roberto-spadim/
