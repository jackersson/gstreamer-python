#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    setup for gstreamer-python package
"""
import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


def read(file):
    return Path(file).read_text('utf-8').strip()


class build_py(_build_py):

    def run(self):
        import subprocess

        def _run_bash_file(bash_file: str):
            if os.path.isfile(bash_file):
                print("Running ... ", bash_file)
                _ = subprocess.run(bash_file, shell=True,
                                   executable="/bin/bash")
            else:
                print("Not found ", bash_file)

        cwd = os.path.dirname(os.path.abspath(__file__))
        _run_bash_file(os.path.join(cwd, 'build-gst-python.sh'))
        _run_bash_file(os.path.join(cwd, 'build-3rd-party.sh'))

        _build_py.run(self)


install_requires = [
    r for r in read('requirements.txt').split('\n') if r]

setup(
    name='gstreamer-python',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="PyGst Utils package",
    long_description='\n\n'.join((read('README.md'))),
    author="LifeStyleTransfer",
    author_email='taras@lifestyletransfer.com',
    url='https://github.com/jackersson/pygst-utils',
    packages=[
        'gstreamer',
    ],
    include_package_data=True,
    install_requires=install_requires,
    license="Apache Software License 2.0",
    zip_safe=True,
    keywords='gstreamer-python',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'build_py': build_py,
    }
)
