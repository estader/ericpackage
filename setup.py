from setuptools import find_packages, setup
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Packages required for this module to be executed
#def list_reqs(fname='requirements.txt'):
#    with open(fname) as fd:
#        return fd.read().splitlines()
    
# Read the requirements
source_root = Path(".")
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()


setup(
    name="ericpackage", 
    version="0.0.1",
    author="Eric Aderne",
    author_email="eeaderne@gmail.com",
    description="Funções da construção de um pipeline em projetos com Aprendizado e Máquina",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:estader/ericpackage",
    install_requires=requirements,
    packages=['ericpackage'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
