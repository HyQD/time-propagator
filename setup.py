from setuptools import setup, find_packages

setup(
    name="time-propagator0",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["qcelemental",
                      "tqdm"],
)

