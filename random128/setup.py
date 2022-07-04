from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include


setup(name="random128",
      ext_modules=cythonize("random128.pyx"),
      include_dirs=[get_include(), "."])