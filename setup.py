import setuptools

setuptools.setup(
    name="AstarTrellis",
    version="0.0.1",
    description="A star algorithm implemented on a trellis data structure to find maximum likelihood hierarchical clustering",
    url="https://github.com/SebastianMacaluso/AstarTrellis",
    author="Nicholas Monath, Craig Greenberg, Sebastian Macaluso",
    author_email="sm4511@nyu.edu",
    license="MIT",
    # packages=setuptools.find_packages(),
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
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
