from setuptools import setup

setup(
    name="taxi_rl_env",
    version="0.0.1",
    description="Customised environment for Open AI Taxi",
    url="https://github.com/cs4246/taxi-rl-env",
    author="Nishita Dutta",
    author_email="nishita.dutta@u.nus.edu",
    packages=["taxi_rl_env"],
    install_requires=["pyzmq", "gym"],
    python_requires=">=3.4",
    setup_requires=['wheel'],
    zip_safe=False,
)