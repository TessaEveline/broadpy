from setuptools import setup, find_packages

setup(
    name="broadpy",
    version="0.1.0",
    description="A Python package for broadening of spectral lines",
    author="Darío González Picos",
    author_email="picos@strw.leidenuniv.nl",
    url="https://github.com/DGonzalezPicos/broadpy",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
