from setuptools import setup, find_packages

setup(
    name="racecar_gym",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.22",
        "pybullet>=3.2.5",
        "nptyping>=1.4.4",
        "yamldataclassconfig>=1.5",
        "pettingzoo>=1.22",
    ],
    python_requires=">=3.10",
    author='Axel Brunnbauer',
    author_email='axel.brunnbauer@gmx.at',
    description='An RL environment for a miniature racecar using the pybullet physics engine.',
    url='https://github.com/axelbr/racecar_gym',
)