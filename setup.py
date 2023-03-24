from setuptools import setup, find_packages
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()


setup(
    name='Activity-Decomp',
    version='0.1',
    url='https://github.com/JMU-Wearable-Computing/activity_decomp',
    author='Riley White',
    author_email='rileywhite89@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires
    )