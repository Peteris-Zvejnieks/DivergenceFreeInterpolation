import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Divergence_Free_Interpolant",
    version = "0.1.5",
    author = "Peteris Zvejnieks",
    author_email = "peterf610@gmail.com",
    description = ("Divergence free interpolant for 2D and 3D systems"),
    license = "MIT",
    keywords = "Interpolation Vector-Field Radial-Basis-Functions",
    url = "https://github.com/Peteris-Zvejnieks/DivergenceFreeInterpolation",
    packages=['Divergence_Free_Interpolant'],
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    setup_requires = ["setuptools", "numpy", "scipy", "sympy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
