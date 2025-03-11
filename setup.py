from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="text-mining-tool",
    version="0.1.0",
    author="JustYuh",
    author_email="justyuh@example.com",
    description="A powerful and flexible tool for extracting insights from text data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JustYuh/text-mining-tool",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "web": [
            "flask>=2.3.0",
            "flask-wtf>=1.1.0",
            "dash>=2.13.0",
            "dash-bootstrap-components>=1.4.0",
        ],
        "nlp": [
            "spacy>=3.6.0",
            "transformers>=4.30.0",
            "gensim>=4.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-mining-tool=src.main:main",
        ],
    },
) 