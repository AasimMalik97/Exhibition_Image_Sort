from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("optimized_code.pyx", language_level="3"),
)
