# Libraries
from setuptools import setup
from Cython.Build import cythonize
import numpy  


# Run in Terminal: 'python setup.py build_ext --inplace'
setup(
    ext_modules=cythonize(["model_ensemble_learning.pyx"]),
    include_dirs=[numpy.get_include()]
)
