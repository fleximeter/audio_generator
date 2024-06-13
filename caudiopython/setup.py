from setuptools import setup, Extension
from distutils import sysconfig
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("analysis", ["analysis.py"], include_dirs=[np.get_include()]),
    Extension("granulator", ["granulator.py"], include_dirs=[np.get_include()]),
    Extension("operations", ["operations.py"], include_dirs=[np.get_include()]),
    Extension("sampler", ["sampler.py"], include_dirs=[np.get_include()]),
    Extension("spectrum", ["spectrum.py"], include_dirs=[np.get_include()]),
]

# courtesy of 
# https://stackoverflow.com/questions/47770000/set-default-compiler-when-using-cython-and-setuptools-to-compile-multiple-extens
# again, an issue with Argon. You probably want to comment this section out if you're not running on Argon.
sysconfig.get_config_vars()['CFLAGS'] = ''
sysconfig.get_config_vars()['OPT'] = ''
sysconfig.get_config_vars()['PY_CFLAGS'] = ''
sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
sysconfig.get_config_vars()['CC'] = 'gcc'
sysconfig.get_config_vars()['CXX'] = 'g++'
sysconfig.get_config_vars()['BASECFLAGS'] = ''
sysconfig.get_config_vars()['CCSHARED'] = '-fPIC'
sysconfig.get_config_vars()['LDSHARED'] = 'gcc -shared'
sysconfig.get_config_vars()['CPP'] = ''
sysconfig.get_config_vars()['CPPFLAGS'] = ''
sysconfig.get_config_vars()['BLDSHARED'] = ''
sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
sysconfig.get_config_vars()['LDFLAGS'] = ''
sysconfig.get_config_vars()['PY_LDFLAGS'] = ''
setup(name="audiopython", ext_modules=cythonize(extensions))
