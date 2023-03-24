from setuptools import setup, find_packages
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Try to import joints, if joints can be successfully 
# imported then we will skip installing it.
try: 
    import joints
    install_requires = [line for line in install_requires if not line.startswith("joint")]
except:
    pass


setup(
    name='activity-decomp',
    version='0.1',
    url='https://github.com/JMU-Wearable-Computing/activity_decomp',
    author='Riley White',
    author_email='rileywhite89@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires
    )