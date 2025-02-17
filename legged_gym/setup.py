from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.2',
    packages=find_packages(),
    description='Go2',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'tensorboard',
                      'tensorboardX',
                      'debugpy']
)