# Advanced Text Mining Tool

A sophisticated text mining and natural language processing (NLP) application designed to extract valuable insights from text data. The tool is designed with a modular architecture that allows it to run with minimal dependencies while providing advanced features when additional packages are installed.

## Features

- **Text Processing & Cleaning**: Advanced preprocessing pipeline for text normalization, tokenization, and noise removal
- **Sentiment Analysis**: Multi-level sentiment classification using state-of-the-art transformer models (requires transformers)
- **Named Entity Recognition**: Identification and extraction of entities such as people, organizations, locations, etc. (requires spaCy)
- **Topic Modeling**: Discover hidden themes in document collections using LDA and BERTopic (requires gensim)
- **Text Classification**: Machine learning models for categorizing documents by content (requires scikit-learn)
- **Text Summarization**: Generate concise summaries of lengthy documents (requires transformers)
- **Keyword Extraction**: Identify important terms and phrases using TF-IDF and RAKE algorithms
- **Visualization**: Interactive dashboards with word clouds, trend graphs, network maps, and more (requires matplotlib, plotly)
- **Multi-format Support**: Process text from CSV, TXT, PDF, DOCX, and web sources (requires pandas, PyPDF2, python-docx, beautifulsoup4)
- **Web Interface**: User-friendly web interface for interactive analysis (requires dash)

## Installation

### Basic Installation (Minimal Dependencies)

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

## Usage

### Command Line Interface

```bash
# Basic usage with minimal dependencies (text preprocessing only)
python src/main.py --input data/sample.txt --output results/

# Run with specific analyses (requires additional dependencies)
python src/main.py --input data/sample.txt --output results/ --analysis sentiment,ner,topics

# Process multiple files
python src/main.py --input data/documents/ --output results/ --format txt

# Start the web interface (requires dash)
python src/app.py
```

### Web Interface

1. Ensure you have installed the web interface dependencies: `pip install dash dash-bootstrap-components`
2. Start the web application: `python src/app.py`
3. Open your browser and navigate to `http://localhost:8050`
4. Upload your documents or enter text directly
5. Select the analyses you want to perform
6. View and export the results

## Project Structure

```
text-mining-tool/
├── src/                    # Source code
│   ├── preprocessing/      # Text cleaning and normalization
│   ├── analysis/           # NLP analysis modules
│   ├── visualization/      # Data visualization components
│   ├── utils/              # Helper functions
│   ├── app.py              # Web application
│   └── main.py             # CLI entry point
├── data/                   # Sample datasets
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── examples/               # Usage examples
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Modular Design

The tool is designed with a modular architecture that allows it to run with minimal dependencies:

- **Core Functionality**: Basic text preprocessing requires only NLTK
- **Optional Modules**: Advanced features are enabled when additional packages are installed
- **Graceful Degradation**: The tool will inform you when a feature is unavailable due to missing dependencies

## Performance Optimization

The tool is optimized for handling large datasets through:
- Parallel processing for CPU-intensive tasks
- Batch processing for memory efficiency
- Caching of intermediate results
- GPU acceleration for transformer models (when available)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project utilizes several open-source NLP libraries including NLTK, spaCy, and Hugging Face Transformers
- Special thanks to the research community for advancing the state of NLP 