#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    setup for pygst-utils package
"""
import os
from pathlib import Path

from setuptools import setup
from setuptools.command.install import install as _install

with open('README.md') as readme_file:
    readme = readme_file.read()


def read(file):
    return Path(file).read_text('utf-8').strip()


class install(_install):

    def run(self):
        import subprocess

        def _run_bash_file(bash_file: str):
            if os.path.isfile(bash_file):
                print("Running ... ", bash_file)
                _ = subprocess.run(bash_file, shell=True, executable="/bin/bash")
            else:
                print("Not found ", bash_file)

        cwd = os.path.dirname(os.path.abspath(__file__))
        _run_bash_file(os.path.join(cwd, 'install_pygst.sh'))
        _run_bash_file(os.path.join(cwd, 'build.sh'))

        _install.run(self)


install_requires = [
    r for r in read('requirements.txt').split('\n') if r]

setup(
    name='pygst_utils',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="PyGst Utils package",
    long_description='\n\n'.join((read('README.md'))),
    author="LifeStyleTransfer",
    author_email='taras.lishchenko@gmail.com',
    url='https://github.com/jackersson/pygst-utils',
    packages=[
        'pygst_utils',
    ],
    include_package_data=True,
    install_requires=install_requires,
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
        'install': install,
    }
)