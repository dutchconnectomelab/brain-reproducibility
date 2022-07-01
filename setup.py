#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="brain-reproducibility",
    version="0.1",
    description='Code for the paper "Reproducibility of neuroimaging studies of brain disorders with hundreds -not thousands- of participants"',
    author="Ilan Libedinsky, Koen Helwegen, Martijn van den Heuvel",
    author_email="k.helwegen@vu.nl",
    packages=find_packages(),
    install_requires=[
        "alive-progress",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "scipy>=1.7.3",
        "statsmodels",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "pytype",
        ],
    },
)
