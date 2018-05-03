from setuptools import setup
from setuptools import find_packages

setup(name='envs-vmpc',
      version='0.0.1',
      author='Varun Tolani',
      author_email='vtolani@berkeley.edu',
      description='Additional environments for RL Seeding experiments',
      packages=find_packages(),
      license='MIT',
      install_requires=['gym']
)
