#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    setup for pygst-utils package
"""
import os

from setuptools import setup
from setuptools.command.build_py import build_py

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'setuptools',
]


class BuildCommand(build_py):
    """
        Build additional packages (./build.sh)
    """
    def run(self):
        import subprocess

        cwd = os.path.dirname(os.path.abspath(__file__))
        build_file = os.path.join(cwd, 'build.sh')
        assert os.path.isfile(build_file), build_file

        _ = subprocess.run(build_file, shell=True, executable="/bin/bash")

        build_py.run(self)

setup(
    name='pygst_utils',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="PyGst Utils package",
    long_description=readme,
    author="LifeStyleTransfer",
    author_email='taras.lishchenko@gmail.com',
    url='https://github.com/jackersson/pygst-utils',
    packages=[
        'pygst_utils',
    ],
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=True,
    keywords='pygst_utils',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'build_py': BuildCommand,
    }
)