# Text Mining Tool

A powerful text analysis tool that provides insights through natural language processing and visualization.

## Features

- **Text Preprocessing**
  - Tokenization
  - Sentence splitting
  - Text cleaning and normalization

- **Analysis**
  - Sentiment Analysis
  - Named Entity Recognition (NER)
  - Keyword Extraction
  - Text Statistics

- **Visualization**
  - Word Clouds
  - Sentiment Distribution Charts
  - Entity Distribution Charts
  - Keyword Frequency Charts

- **Export Options**
  - JSON format
  - CSV format
  - Plain text format
  - Visualization images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TextMiningTool.git
cd TextMiningTool
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"
```

## Usage

### Web Interface

1. Start the web server:
```bash
python src/web/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter or paste your text and click "Process" to analyze.

### API Usage

The tool provides REST API endpoints for integration:

```python
import requests

# Process text
response = requests.post('http://localhost:5000/api/process', 
                        json={'text': 'Your text here'})
results = response.json()

# Specific analysis
response = requests.post('http://localhost:5000/api/analyze',
                        json={'text': 'Your text here', 
                             'analysis_type': 'sentiment'})
sentiment = response.json()
```

## Project Structure

```
TextMiningTool/
├── src/                    # Source code
│   ├── analysis/          # Text analysis modules
│   ├── preprocessing/     # Text preprocessing
│   ├── visualization/     # Visualization tools
│   ├── web/              # Web interface
│   └── utils/            # Utility functions
├── tests/                 # Test files
├── docs/                  # Documentation
├── examples/             # Usage examples
├── data/                 # Sample data
└── sample_results/       # Example outputs
```

## Advanced Features

### Customization

- Modify `src/preprocessing/text_processor.py` for custom text preprocessing
- Add new visualizations in `src/visualization/visualizer.py`
- Extend analysis capabilities in `src/analysis/` modules

### API Endpoints

- `/api/process`: Full text processing
- `/api/analyze`: Specific analysis types
- `/api/visualize`: Generate visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NLTK for natural language processing
- Matplotlib and WordCloud for visualizations
- Flask for the web interface 