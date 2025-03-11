# Web Interface for the Text Mining Tool

This directory contains the web interface for the Text Mining Tool.

## Directory Structure

- `app.py`: The main Flask application
- `templates/`: HTML templates
  - `index.html`: The main page template
  - `results.html`: The results page template
- `static/`: Static files
  - `css/`: CSS stylesheets
  - `js/`: JavaScript files
  - `images/`: Image files

## Running the Web Interface

To run the web interface:

```bash
# Make sure you have the required dependencies
pip install flask flask-wtf

# Run the application
python src/web/app.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Features

The web interface provides the following features:

- Upload text files for analysis
- Enter text directly for analysis
- Select which analyses to perform
- View results in a user-friendly format
- Download results in various formats (JSON, CSV, TXT)
- Visualize results with charts and graphs

## Customization

You can customize the web interface by modifying the templates and static files. The templates use the Jinja2 templating engine, which is included with Flask.

## API Endpoints

The web interface also provides API endpoints for programmatic access:

- `/api/process`: Process text and return results
- `/api/analyze`: Analyze processed text
- `/api/visualize`: Generate visualizations

See the [API documentation](../../docs/api.md) for more information. 