from setuptools import setup, find_packages

setup(name='taxi-rl-env',
      version='0.0.1',
      install_requires=['gym==0.26.2', 'numpy'],
      packages=['taxi_rl_env']
)