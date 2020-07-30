from distutils.core import setup
from Cython.Build import cythonize
import numpy

if __name__ == "__main__":
    setup(
        name='mhp',
        ext_modules=cythonize("cumulant_computer.pyx", gdb_debug=True),
        include_dirs=[numpy.get_include()]
    )
