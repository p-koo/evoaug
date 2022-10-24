from setuptools import setup, find_packages


setup(
    name="evoaug",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["torch"],
)
