import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command


import setuptools

setuptools.setup(
    name="A_Star_Trellis",
    version="0.1",
    description="A Star Trellis",
    url="https://github.com/SebastianMacaluso/a_star_trellis",
    author="Nicholas Monath, Craig Greenberg, Sebastian Macaluso",
    author_email="sm4511@nyu.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
)





#
# #!/usr/bin/env python
#
# from distutils.core import setup
#
# setup(name='a_star_trellis',
#       version='0.01',
#       packages=['a_star_trellis'],
#       install_requires=[
#           "numpy",
#           "absl-py",
#           "python-dateutil",
#           "scipy"
#       ],
#       packages=setuptools.find_packages(),
#       # package_dir={'a_star_trellis': 'a_star_trellis'}
#       )
#
