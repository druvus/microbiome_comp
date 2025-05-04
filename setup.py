from setuptools import setup, find_packages

setup(
    name="microbiome_comp",
    version="0.1.0",
    description=(
        "Microbial‐genomics compositional workflows: "
        "taxa filtering, Bayesian zero‐replacement, and PLR transforms"
    ),
    author="Andreas Sjödin",
    author_email="andreas.sjodinyou@gmail.com",
    url="https://github.com/druvus/microbiome_comp",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.18",
        "pandas>=1.0",
        "scipy>=1.4",
        "numba>=0.48",
        "tqdm>=4.0",
    ],
    entry_points={
        "console_scripts": [
            "microbiome_comp = microbiome_comp.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)
