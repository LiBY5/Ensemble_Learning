# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ensemble-learning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive ensemble learning project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ensemble_learning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
)
