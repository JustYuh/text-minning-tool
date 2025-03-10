#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Text Mining Tool - Web Interface
-----------------------------------------
This module provides a web-based user interface for the text mining tool using Dash.
It allows users to upload documents, select analyses, and view results interactively.
"""

import base64
import datetime
import io
import json
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from src.preprocessing import text_processor
from src.analysis import sentiment, entity_recognition, topic_modeling, text_classifier, summarizer, keyword_extractor
from src.utils import file_handler, logger_config
from src.visualization import visualizer

# Configure logging
logger = logger_config.setup_logger(__name__)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

app.title = "Advanced Text Mining Tool"
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Advanced Text Mining Tool", className="text-center my-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Data"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=True
                            ),
                            html.Div(id='upload-output'),
                        ], label="Upload Files"),
                        dbc.Tab([
                            dbc.Textarea(
                                id='text-input',
                                placeholder="Enter text to analyze...",
                                style={'width': '100%', 'height': '200px'},
                                className="mb-3"
                            ),
                        ], label="Enter Text"),
                        dbc.Tab([
                            dbc.Input(
                                id='url-input',
                                placeholder="Enter URL to scrape (e.g., https://example.com)",
                                type="url",
                                className="mb-3"
                            ),
                            dbc.Button("Fetch Content", id="fetch-url", color="primary", className="mb-3"),
                            html.Div(id='url-output'),
                        ], label="Web Scraping"),
                    ]),
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Analysis Options"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Analyses:"),
                            dbc.Checklist(
                                id='analysis-options',
                                options=[
                                    {"label": "Sentiment Analysis", "value": "sentiment"},
                                    {"label": "Named Entity Recognition", "value": "ner"},
                                    {"label": "Topic Modeling", "value": "topics"},
                                    {"label": "Text Classification", "value": "classify"},
                                    {"label": "Text Summarization", "value": "summarize"},
                                    {"label": "Keyword Extraction", "value": "keywords"},
                                ],
                                value=["sentiment", "ner", "keywords"],
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Language:"),
                            dbc.Select(
                                id='language-select',
                                options=[
                                    {"label": "English", "value": "en"},
                                    {"label": "Spanish", "value": "es"},
                                    {"label": "French", "value": "fr"},
                                    {"label": "German", "value": "de"},
                                    {"label": "Chinese", "value": "zh"},
                                ],
                                value="en",
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Model Size:"),
                            dbc.Select(
                                id='model-select',
                                options=[
                                    {"label": "Default", "value": "default"},
                                    {"label": "Small (Faster)", "value": "small"},
                                    {"label": "Large (More Accurate)", "value": "large"},
                                ],
                                value="default",
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    dbc.Button("Run Analysis", id="run-analysis", color="success", className="mt-2", size="lg", block=True),
                ])
            ], className="mb-4"),
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analysis Results"),
                dbc.CardBody([
                    dbc.Spinner(
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id='sentiment-output')
                            ], label="Sentiment Analysis", tab_id="sentiment-tab"),
                            dbc.Tab([
                                html.Div(id='ner-output')
                            ], label="Named Entities", tab_id="ner-tab"),
                            dbc.Tab([
                                html.Div(id='topics-output')
                            ], label="Topics", tab_id="topics-tab"),
                            dbc.Tab([
                                html.Div(id='classification-output')
                            ], label="Classification", tab_id="classification-tab"),
                            dbc.Tab([
                                html.Div(id='summary-output')
                            ], label="Summary", tab_id="summary-tab"),
                            dbc.Tab([
                                html.Div(id='keywords-output')
                            ], label="Keywords", tab_id="keywords-tab"),
                            dbc.Tab([
                                html.Div(id='visualization-output')
                            ], label="Visualizations", tab_id="visualization-tab"),
                        ], id="results-tabs", active_tab="sentiment-tab"),
                        color="primary",
                    ),
                ])
            ], className="h-100"),
        ], width=8),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Advanced Text Mining Tool © 2025", className="text-center text-muted"),
        ], width=12)
    ]),
    
    # Store components for intermediate data
    dcc.Store(id='processed-data'),
    dcc.Store(id='analysis-results'),
], fluid=True)

# Callback for file upload
@app.callback(
    Output('upload-output', 'children'),
    Output('processed-data', 'data', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        return html.Div("No files uploaded."), None
    
    children = []
    all_text = []
    
    for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            
            try:
                if 'csv' in name:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    text = '\n'.join(df.iloc[:, 0].astype(str).tolist())
                elif 'xls' in name:
                    df = pd.read_excel(io.BytesIO(decoded))
                    text = '\n'.join(df.iloc[:, 0].astype(str).tolist())
                elif 'txt' in name or 'md' in name:
                    text = decoded.decode('utf-8')
                elif 'pdf' in name:
                    # This would use PyPDF2 in a real implementation
                    text = f"PDF content from {name} (placeholder)"
                elif 'docx' in name:
                    # This would use python-docx in a real implementation
                    text = f"DOCX content from {name} (placeholder)"
                else:
                    text = decoded.decode('utf-8')
                
                all_text.append(text)
                
                children.append(
                    dbc.ListGroupItem([
                        html.H5(name),
                        html.P(f"Last modified: {datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"),
                        html.P(f"Size: {len(decoded) / 1024:.2f} KB"),
                    ])
                )
            except Exception as e:
                children.append(
                    dbc.ListGroupItem([
                        html.H5(name, style={"color": "red"}),
                        html.P(f"Error processing file: {str(e)}"),
                    ])
                )
        except Exception as e:
            children.append(
                dbc.ListGroupItem([
                    html.H5("Error", style={"color": "red"}),
                    html.P(f"Error processing upload: {str(e)}"),
                ])
            )
    
    return dbc.ListGroup(children), json.dumps(all_text)

# Callback for text input
@app.callback(
    Output('processed-data', 'data'),
    Input('text-input', 'value'),
    prevent_initial_call=True
)
def process_text_input(text):
    if text is None or text.strip() == "":
        return None
    return json.dumps([text])

# Callback for URL fetching
@app.callback(
    Output('url-output', 'children'),
    Output('processed-data', 'data', allow_duplicate=True),
    Input('fetch-url', 'n_clicks'),
    State('url-input', 'value'),
    prevent_initial_call=True
)
def fetch_url_content(n_clicks, url):
    if n_clicks is None or url is None or url.strip() == "":
        return html.Div("No URL provided."), None
    
    try:
        # In a real implementation, this would use requests and BeautifulSoup
        # to fetch and parse the content
        text = f"Content from {url} (placeholder)"
        
        return html.Div([
            html.P(f"Successfully fetched content from {url}"),
            html.P(f"Content length: {len(text)} characters"),
        ]), json.dumps([text])
    except Exception as e:
        return html.Div(f"Error fetching URL: {str(e)}"), None

# Callback for running analysis
@app.callback(
    Output('analysis-results', 'data'),
    Input('run-analysis', 'n_clicks'),
    State('processed-data', 'data'),
    State('analysis-options', 'value'),
    State('language-select', 'value'),
    State('model-select', 'value'),
    prevent_initial_call=True
)
def run_analysis(n_clicks, data, analyses, language, model_size):
    if n_clicks is None or data is None:
        return None
    
    try:
        # Parse the JSON data
        documents = json.loads(data)
        
        if not documents:
            return json.dumps({"error": "No documents to analyze."})
        
        # Initialize the text processor
        processor = text_processor.TextProcessor(language=language)
        processed_docs = processor.process_batch(documents)
        
        results = {}
        
        # Run the selected analyses
        if 'sentiment' in analyses:
            sentiment_analyzer = sentiment.SentimentAnalyzer(model_size=model_size)
            results['sentiment'] = sentiment_analyzer.analyze_batch(processed_docs)
        
        if 'ner' in analyses:
            ner = entity_recognition.NamedEntityRecognizer(model_size=model_size)
            results['entities'] = ner.extract_entities_batch(processed_docs)
        
        if 'topics' in analyses:
            topic_modeler = topic_modeling.TopicModeler(n_topics=5)
            results['topics'] = topic_modeler.extract_topics(processed_docs)
        
        if 'classify' in analyses:
            classifier = text_classifier.TextClassifier(model_size=model_size)
            results['categories'] = classifier.classify_batch(processed_docs)
        
        if 'summarize' in analyses:
            text_summarizer = summarizer.TextSummarizer(model_size=model_size)
            results['summaries'] = text_summarizer.summarize_batch(processed_docs)
        
        if 'keywords' in analyses:
            extractor = keyword_extractor.KeywordExtractor()
            results['keywords'] = extractor.extract_keywords_batch(processed_docs)
        
        # Add the original and processed text to the results
        results['original_text'] = documents
        results['processed_text'] = processed_docs
        
        return json.dumps(results)
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return json.dumps({"error": str(e)})

# Callback for sentiment analysis output
@app.callback(
    Output('sentiment-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_sentiment_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'sentiment' not in results:
            return html.Div("Sentiment analysis was not performed.")
        
        sentiment_results = results['sentiment']
        
        # Create a sentiment visualization
        fig = visualizer.create_sentiment_chart(sentiment_results)
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H5("Sentiment Details"),
            html.Div([
                html.P(f"Document {i+1}: {result['label']} ({result['score']:.2f})")
                for i, result in enumerate(sentiment_results)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering sentiment results: {str(e)}")

# Callback for named entity recognition output
@app.callback(
    Output('ner-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_ner_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'entities' not in results:
            return html.Div("Named entity recognition was not performed.")
        
        entity_results = results['entities']
        
        # Create entity visualizations
        entity_chart = visualizer.create_entity_chart(entity_results)
        entity_network = visualizer.create_entity_network(entity_results)
        
        return html.Div([
            dcc.Graph(figure=entity_chart),
            html.Hr(),
            dcc.Graph(figure=entity_network),
            html.Hr(),
            html.H5("Entity Details"),
            html.Div([
                html.Div([
                    html.H6(f"Document {i+1}"),
                    html.Ul([
                        html.Li(f"{entity['text']} ({entity['label']})")
                        for entity in doc_entities
                    ])
                ])
                for i, doc_entities in enumerate(entity_results)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering entity results: {str(e)}")

# Callback for topic modeling output
@app.callback(
    Output('topics-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_topics_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'topics' not in results:
            return html.Div("Topic modeling was not performed.")
        
        topic_results = results['topics']
        
        # Create topic visualizations
        topic_chart = visualizer.create_topic_chart(topic_results)
        
        return html.Div([
            dcc.Graph(figure=topic_chart),
            html.Hr(),
            html.H5("Topic Details"),
            html.Div([
                html.Div([
                    html.H6(f"Topic {i+1}"),
                    html.P(", ".join(topic['words'])),
                    html.P(f"Weight: {topic['weight']:.2f}")
                ])
                for i, topic in enumerate(topic_results['topics'])
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering topic results: {str(e)}")

# Callback for classification output
@app.callback(
    Output('classification-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_classification_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'categories' not in results:
            return html.Div("Text classification was not performed.")
        
        category_results = results['categories']
        
        # Create classification visualization
        category_chart = visualizer.create_category_chart(category_results)
        
        return html.Div([
            dcc.Graph(figure=category_chart),
            html.Hr(),
            html.H5("Classification Details"),
            html.Div([
                html.P(f"Document {i+1}: {result['label']} ({result['score']:.2f})")
                for i, result in enumerate(category_results)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering classification results: {str(e)}")

# Callback for summary output
@app.callback(
    Output('summary-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_summary_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'summaries' not in results:
            return html.Div("Text summarization was not performed.")
        
        summary_results = results['summaries']
        
        return html.Div([
            html.H5("Document Summaries"),
            html.Div([
                html.Div([
                    html.H6(f"Document {i+1}"),
                    html.P(summary),
                    html.P(f"Compression Ratio: {len(summary) / len(results['original_text'][i]):.2f}")
                ])
                for i, summary in enumerate(summary_results)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering summary results: {str(e)}")

# Callback for keyword output
@app.callback(
    Output('keywords-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_keywords_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        if 'keywords' not in results:
            return html.Div("Keyword extraction was not performed.")
        
        keyword_results = results['keywords']
        
        # Create keyword visualization
        keyword_chart = visualizer.create_keyword_chart(keyword_results)
        
        # Generate word cloud image
        wordcloud_img = visualizer.create_wordcloud(keyword_results)
        
        return html.Div([
            html.Img(src=wordcloud_img, style={'width': '100%'}),
            html.Hr(),
            dcc.Graph(figure=keyword_chart),
            html.Hr(),
            html.H5("Keyword Details"),
            html.Div([
                html.Div([
                    html.H6(f"Document {i+1}"),
                    html.Ul([
                        html.Li(f"{keyword['text']} (score: {keyword['score']:.2f})")
                        for keyword in doc_keywords
                    ])
                ])
                for i, doc_keywords in enumerate(keyword_results)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering keyword results: {str(e)}")

# Callback for visualization output
@app.callback(
    Output('visualization-output', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True
)
def update_visualization_output(data):
    if data is None:
        return html.Div("No analysis results available.")
    
    try:
        results = json.loads(data)
        
        if 'error' in results:
            return html.Div(f"Error: {results['error']}")
        
        # Create a dashboard with multiple visualizations
        visualizations = []
        
        if 'sentiment' in results:
            sentiment_chart = visualizer.create_sentiment_chart(results['sentiment'])
            visualizations.append(
                dbc.Card([
                    dbc.CardHeader("Sentiment Analysis"),
                    dbc.CardBody([
                        dcc.Graph(figure=sentiment_chart)
                    ])
                ])
            )
        
        if 'entities' in results:
            entity_chart = visualizer.create_entity_chart(results['entities'])
            visualizations.append(
                dbc.Card([
                    dbc.CardHeader("Named Entities"),
                    dbc.CardBody([
                        dcc.Graph(figure=entity_chart)
                    ])
                ])
            )
        
        if 'topics' in results:
            topic_chart = visualizer.create_topic_chart(results['topics'])
            visualizations.append(
                dbc.Card([
                    dbc.CardHeader("Topic Distribution"),
                    dbc.CardBody([
                        dcc.Graph(figure=topic_chart)
                    ])
                ])
            )
        
        if 'keywords' in results:
            wordcloud_img = visualizer.create_wordcloud(results['keywords'])
            visualizations.append(
                dbc.Card([
                    dbc.CardHeader("Word Cloud"),
                    dbc.CardBody([
                        html.Img(src=wordcloud_img, style={'width': '100%'})
                    ])
                ])
            )
        
        if not visualizations:
            return html.Div("No visualizations available. Run analyses to generate visualizations.")
        
        return html.Div([
            dbc.Row([
                dbc.Col(viz, width=6, className="mb-4")
                for viz in visualizations
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error rendering visualizations: {str(e)}")

if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 