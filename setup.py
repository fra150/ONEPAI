# -----------------------------------------------------------------------------
# File: setup.py
# -----------------------------------------------------------------------------

"""Configurazione per l'installazione del pacchetto ONEPAI."""

from setuptools import find_packages, setup

# Leggi il contenuto di README.md per la long_description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Leggi le dipendenze da requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="onepai",
    version="0.1.0-alpha",
    author="Brainverse",
    author_email="francesco.bulla@brainverse.it",
    description="ONEPAI - Il Tesoro dell'AI: L'intelligenza dell'invisibile.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brainverse/onepai",
    project_urls={
        "Bug Tracker": "https://github.com/brainverse/onepai/issues",
        "Source Code": "https://github.com/brainverse/onepai",
    },
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords="ai, interpretability, explainability, neural-networks, pytorch, tensorflow",
)