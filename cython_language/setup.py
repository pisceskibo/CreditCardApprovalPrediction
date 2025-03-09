# Libraries
from setuptools import setup
from Cython.Build import cythonize
import numpy  


# Run in Terminal: 'python setup.py build_ext --inplace'
setup(
    ext_modules=cythonize(["data_analysis.pyx", "data_preprocessing.pyx", "data_prediction.pyx"]),
    include_dirs=[numpy.get_include()]
)
