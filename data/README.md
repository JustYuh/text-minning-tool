# Data for the Text Mining Tool

This directory contains sample data files for the Text Mining Tool.

## Sample Files

- `sample.txt`: A sample text file about text mining

## Adding Your Own Data

You can add your own data files to this directory. The tool supports the following file formats:

- `.txt`: Plain text files
- `.csv`: CSV files with text data
- `.json`: JSON files with text data
- `.pdf`: PDF files (requires PyPDF2)
- `.docx`: Word documents (requires python-docx)
- `.html`: HTML files (requires beautifulsoup4)

## Using Data Files

To process a data file, use the command-line interface:

```bash
python src/main.py --input data/sample.txt --output results/
```

Or use the web interface:

1. Start the web application: `python src/app.py`
2. Open your browser and navigate to `http://localhost:8050`
3. Upload your data file
4. Select the analyses you want to perform
5. View and export the results 