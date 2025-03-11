# Getting Started with the Text Mining Tool

This guide will help you get started with the Text Mining Tool, a powerful application for extracting insights from text data.

## Prerequisites

- Python 3.8 or higher
- Basic knowledge of command-line interfaces

## Installation

### Basic Installation (Minimal Dependencies)

If you only need basic text preprocessing capabilities, you can install the minimal dependencies:

```bash
# Clone the repository
git clone https://github.com/JustYuh/text-minning-tool.git
cd text-minning-tool

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies
pip install nltk

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Full Installation (All Features)

For access to all features, install the full set of dependencies:

```bash
# Clone the repository
git clone https://github.com/JustYuh/text-minning-tool.git
cd text-minning-tool

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy models
python -m spacy download en_core_web_sm
```

## Basic Usage

### Command Line Interface

The tool provides a command-line interface for processing text files:

```bash
# Basic usage with minimal dependencies (text preprocessing only)
python src/main.py --input data/sample.txt --output results/

# Run with specific analyses (requires additional dependencies)
python src/main.py --input data/sample.txt --output results/ --analysis sentiment,ner,topics

# Process multiple files
python src/main.py --input data/documents/ --output results/ --format txt
```

### Web Interface

The tool also provides a web interface for interactive analysis:

1. Ensure you have installed the web interface dependencies: `pip install dash dash-bootstrap-components`
2. Start the web application: `python src/app.py`
3. Open your browser and navigate to `http://localhost:8050`
4. Upload your documents or enter text directly
5. Select the analyses you want to perform
6. View and export the results

## Example Scripts

The `examples` directory contains example scripts that demonstrate how to use the tool:

- `basic_usage.py`: Demonstrates how to use the tool with minimal dependencies
- `advanced_usage.py`: Shows how to use advanced features when additional dependencies are installed
- `nltk_mining.py`: A standalone script for text mining using NLTK

To run an example script:

```bash
python examples/basic_usage.py
```

## Next Steps

- Check out the [README.md](../README.md) file for more information about the tool
- Explore the [examples](../examples) directory for more usage examples
- Read the [API documentation](api.md) for details about the tool's API 