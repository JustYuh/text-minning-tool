# Source Code Documentation

## Module Structure

### analysis/
- `sentiment.py`: Sentiment analysis using NLTK's VADER
- `entities.py`: Named Entity Recognition using NLTK's NE Chunker
- `keywords.py`: Keyword extraction using TF-IDF and frequency analysis

### preprocessing/
- `text_processor.py`: Text cleaning, tokenization, and normalization
- `tokenizer.py`: Custom tokenization rules and functions
- `normalizer.py`: Text normalization utilities

### visualization/
- `visualizer.py`: Visualization generation functions
- `charts.py`: Chart creation utilities using Matplotlib
- `wordcloud_generator.py`: Word cloud generation

### web/
- `app.py`: Flask web application and API endpoints
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and static assets

### utils/
- `file_handler.py`: File I/O operations
- `text_utils.py`: Text manipulation utilities
- `config.py`: Configuration settings

## Key Components

### Text Processing Pipeline
1. Text input validation and cleaning
2. Tokenization and sentence splitting
3. Normalization and preprocessing
4. Analysis (sentiment, entities, keywords)
5. Visualization generation
6. Result formatting and export

### Web Interface
- RESTful API endpoints
- File upload handling
- Result visualization
- Export functionality

### Visualization Features
- Sentiment distribution charts
- Entity frequency charts
- Keyword importance charts
- Word clouds

## Development Guidelines

### Adding New Features
1. Create a new module in the appropriate directory
2. Update the corresponding __init__.py file
3. Add tests in the tests/ directory
4. Update documentation

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function arguments
- Include docstrings for all functions
- Keep functions focused and modular

### Error Handling
- Use try-except blocks for external operations
- Provide meaningful error messages
- Log errors appropriately
- Implement graceful fallbacks

### Testing
- Write unit tests for new features
- Update integration tests as needed
- Test edge cases and error conditions
- Verify visualization outputs

## Dependencies

### Required
- Flask==3.0.2
- nltk==3.8.1
- numpy==1.26.4
- pandas==2.2.1

### Optional
- matplotlib==3.10.1
- wordcloud==1.9.4
- plotly==6.0.0
- seaborn==0.13.2

## Configuration

Key configuration files:
- `config.py`: General settings
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (create from .env.example)

## Troubleshooting

Common issues and solutions:
1. NLTK Data Missing
   ```python
   import nltk
   nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
   ```

2. Visualization Errors
   - Ensure matplotlib is installed
   - Check write permissions for output directory
   - Verify seaborn installation

3. Web Server Issues
   - Check port availability (default: 5000)
   - Verify Flask installation
   - Check static file permissions 