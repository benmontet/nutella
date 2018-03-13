#!/usr/bin/env python
import sys
from setuptools import setup
# To use a consistent encoding
from codecs import open
import os
from os import path

# Load the version variable
exec(open('nutella/version.py').read())

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/nutella*")
    sys.exit()

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nutella',
    packages=['nutella'],
    version='0.0.0',
    description='great (point) spreads for Kepler, K2, and TESS',
    long_description=long_description,
    url='https://github.com/benmontet/nutella',
    author='Ben Montet and GitHub contributors',
    author_email='bmontet@uchicago.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    keywords='astronomy statistics probability',
    install_requires=['numpy', 'scipy', 'vaneska']
)
