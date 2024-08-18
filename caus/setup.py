from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("analysis", ["analysis.py"], include_dirs=[np.get_include()]),
    Extension("granulator", ["granulator.py"], include_dirs=[np.get_include()]),
    Extension("operations", ["operations.py"], include_dirs=[np.get_include()]),
    Extension("sampler", ["sampler.py"], include_dirs=[np.get_include()]),
    Extension("spectrum", ["spectrum.py"], include_dirs=[np.get_include()]),
    Extension("synthesis", ["synthesis.py"], include_dirs=[np.get_include()]),
]

setup(name="aus", ext_modules=cythonize(extensions))
