#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Module
------------------
This module provides functions for visualizing text mining results,
including sentiment analysis, named entity recognition, topic modeling,
and keyword extraction.
"""

import base64
import io
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter
import os
import random

# Import visualization dependencies with availability checks
from . import MATPLOTLIB_AVAILABLE, WORDCLOUD_AVAILABLE, PLOTLY_AVAILABLE

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    import seaborn as sns

if WORDCLOUD_AVAILABLE:
    from wordcloud import WordCloud

if PLOTLY_AVAILABLE:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)

# Set default color schemes
COLOR_SCHEMES = {
    'sentiment': {
        'positive': '#2ecc71',  # Green
        'neutral': '#3498db',   # Blue
        'negative': '#e74c3c'   # Red
    },
    'entity_types': {
        'PERSON': '#3498db',      # Blue
        'ORGANIZATION': '#2ecc71', # Green
        'LOCATION': '#9b59b6',     # Purple
        'DATE': '#f1c40f',         # Yellow
        'TIME': '#e67e22',         # Orange
        'MONEY': '#27ae60',        # Dark Green
        'PERCENT': '#16a085',      # Teal
        'PRODUCT': '#e74c3c',      # Red
        'EVENT': '#d35400',        # Dark Orange
        'WORK_OF_ART': '#8e44ad',  # Dark Purple
        'LAW': '#2980b9',          # Dark Blue
        'LANGUAGE': '#1abc9c',     # Light Teal
        'GROUP': '#f39c12',        # Dark Yellow
        'MISCELLANEOUS': '#7f8c8d', # Gray
        'OTHER': '#95a5a6'         # Light Gray
    }
}

def create_sentiment_chart(sentiment_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a visualization of sentiment analysis results.
    
    Args:
        sentiment_results: A list of sentiment analysis results.
        
    Returns:
        A Plotly figure object.
    """
    if not sentiment_results:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Count sentiments
    sentiment_counts = Counter([result['label'] for result in sentiment_results])
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    # Get colors for each sentiment
    colors = [COLOR_SCHEMES['sentiment'].get(label, '#95a5a6') for label in labels]
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    # Update layout
    fig.update_layout(
        title="Sentiment Distribution",
        legend_title="Sentiment",
        height=500
    )
    
    return fig

def create_entity_chart(entity_results: List[List[Dict[str, Any]]]) -> go.Figure:
    """
    Create a visualization of named entity recognition results.
    
    Args:
        entity_results: A list of lists of entity dictionaries.
        
    Returns:
        A Plotly figure object.
    """
    if not entity_results or all(not entities for entities in entity_results):
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No entity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Flatten the list of lists
    all_entities = []
    for doc_entities in entity_results:
        all_entities.extend(doc_entities)
    
    # Count entity types
    entity_type_counts = Counter([entity['label'] for entity in all_entities])
    
    # Sort by count
    sorted_types = sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)
    entity_types = [item[0] for item in sorted_types]
    counts = [item[1] for item in sorted_types]
    
    # Get colors for each entity type
    colors = [COLOR_SCHEMES['entity_types'].get(entity_type, '#95a5a6') for entity_type in entity_types]
    
    # Create a bar chart
    fig = go.Figure(data=[go.Bar(
        x=entity_types,
        y=counts,
        marker=dict(color=colors)
    )])
    
    # Update layout
    fig.update_layout(
        title="Named Entity Types",
        xaxis_title="Entity Type",
        yaxis_title="Count",
        height=500
    )
    
    return fig

def create_entity_network(entity_results: List[List[Dict[str, Any]]], max_entities: int = 50) -> go.Figure:
    """
    Create a network visualization of entity co-occurrences.
    
    Args:
        entity_results: A list of lists of entity dictionaries.
        max_entities: The maximum number of entities to include in the network.
        
    Returns:
        A Plotly figure object.
    """
    if not entity_results or all(not entities for entities in entity_results):
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No entity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a graph
    G = nx.Graph()
    
    # Process each document
    for doc_entities in entity_results:
        # Skip empty documents
        if not doc_entities:
            continue
        
        # Sort entities by position
        sorted_entities = sorted(doc_entities, key=lambda x: x.get('start', 0))
        
        # Add nodes and edges
        for i, entity1 in enumerate(sorted_entities[:-1]):
            # Add the entity as a node if it doesn't exist
            if entity1['text'] not in G:
                G.add_node(entity1['text'], type=entity1['label'])
            
            # Connect with nearby entities (window size of 5)
            for j in range(i+1, min(i+6, len(sorted_entities))):
                entity2 = sorted_entities[j]
                
                # Add the second entity as a node if it doesn't exist
                if entity2['text'] not in G:
                    G.add_node(entity2['text'], type=entity2['label'])
                
                # Add or update the edge
                if G.has_edge(entity1['text'], entity2['text']):
                    G[entity1['text']][entity2['text']]['weight'] += 1
                else:
                    G.add_edge(entity1['text'], entity2['text'], weight=1)
    
    # If the graph is empty, return an empty figure
    if not G.nodes():
        fig = go.Figure()
        fig.add_annotation(
            text="No entity co-occurrences found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Limit the number of entities if there are too many
    if len(G.nodes()) > max_entities:
        # Keep only the nodes with the highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_entities]
        top_node_names = [node[0] for node in top_nodes]
        
        # Create a subgraph with only the top nodes
        G = G.subgraph(top_node_names)
    
    # Use a layout algorithm to position the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight, color='rgba(150,150,150,0.5)'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    
    # Create node traces for each entity type
    node_traces = {}
    
    for node in G.nodes(data=True):
        node_name = node[0]
        node_type = node[1]['type']
        x, y = pos[node_name]
        
        # Create a trace for this entity type if it doesn't exist
        if node_type not in node_traces:
            node_traces[node_type] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color=COLOR_SCHEMES['entity_types'].get(node_type, '#95a5a6'),
                    size=10,
                    line=dict(width=1, color='white')
                ),
                name=node_type
            )
        
        # Add the node to the trace
        node_traces[node_type]['x'] = node_traces[node_type]['x'] + (x,)
        node_traces[node_type]['y'] = node_traces[node_type]['y'] + (y,)
        node_traces[node_type]['text'] = node_traces[node_type]['text'] + (f"{node_name} ({node_type})",)
    
    # Create the figure
    fig = go.Figure(data=edge_traces + list(node_traces.values()))
    
    # Update layout
    fig.update_layout(
        title="Entity Co-occurrence Network",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_topic_chart(topic_results: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization of topic modeling results.
    
    Args:
        topic_results: A dictionary containing topic modeling results.
        
    Returns:
        A Plotly figure object.
    """
    if not topic_results or 'topics' not in topic_results or not topic_results['topics']:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No topic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    topics = topic_results['topics']
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Topic Weights", "Top Words by Topic"),
        vertical_spacing=0.2,
        row_heights=[0.3, 0.7]
    )
    
    # Add topic weights bar chart
    topic_ids = [f"Topic {topic['id'] + 1}" for topic in topics]
    topic_weights = [topic['weight'] for topic in topics]
    
    fig.add_trace(
        go.Bar(
            x=topic_ids,
            y=topic_weights,
            marker=dict(color='rgba(50, 171, 96, 0.7)'),
            name="Topic Weight"
        ),
        row=1, col=1
    )
    
    # Add top words heatmap
    topic_words = []
    word_weights = []
    
    for topic in topics:
        # Get the top 10 words for this topic
        words = topic['words'][:10]
        weights = topic['weights'][:10]
        
        topic_words.append(words)
        word_weights.append(weights)
    
    # Create a heatmap of word weights
    heatmap_data = []
    for i, (words, weights) in enumerate(zip(topic_words, word_weights)):
        for j, (word, weight) in enumerate(zip(words, weights)):
            heatmap_data.append({
                'Topic': f"Topic {i + 1}",
                'Word': word,
                'Weight': weight
            })
    
    df = pd.DataFrame(heatmap_data)
    pivot_df = df.pivot(index='Topic', columns='Word', values='Weight')
    
    # Fill NaN values with 0
    pivot_df = pivot_df.fillna(0)
    
    # Create the heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            colorbar=dict(title='Weight'),
            hoverongaps=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Topic Modeling Results",
        height=800,
        showlegend=False
    )
    
    return fig

def create_category_chart(category_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a visualization of text classification results.
    
    Args:
        category_results: A list of classification results.
        
    Returns:
        A Plotly figure object.
    """
    if not category_results:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No classification data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Count categories
    category_counts = Counter([result['label'] for result in category_results])
    
    # Sort by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]
    
    # Create a bar chart
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=counts,
        marker=dict(color='rgba(50, 171, 96, 0.7)')
    )])
    
    # Update layout
    fig.update_layout(
        title="Document Categories",
        xaxis_title="Category",
        yaxis_title="Count",
        height=500
    )
    
    return fig

def create_keyword_chart(keyword_results: List[List[Dict[str, Any]]], top_n: int = 20) -> go.Figure:
    """
    Create a visualization of keyword extraction results.
    
    Args:
        keyword_results: A list of lists of keyword dictionaries.
        top_n: The number of top keywords to display.
        
    Returns:
        A Plotly figure object.
    """
    if not keyword_results or all(not keywords for keywords in keyword_results):
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No keyword data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Flatten the list of lists and count keywords
    all_keywords = []
    for doc_keywords in keyword_results:
        all_keywords.extend(doc_keywords)
    
    keyword_counts = Counter([keyword['text'] for keyword in all_keywords])
    
    # Get the top N keywords
    top_keywords = keyword_counts.most_common(top_n)
    keywords = [item[0] for item in top_keywords]
    counts = [item[1] for item in top_keywords]
    
    # Create a horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        y=keywords,
        x=counts,
        orientation='h',
        marker=dict(
            color='rgba(50, 171, 96, 0.7)',
            line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=f"Top {len(keywords)} Keywords",
        xaxis_title="Frequency",
        yaxis_title="Keyword",
        height=max(500, 20 * len(keywords))
    )
    
    return fig

def create_wordcloud(keyword_results: List[List[Dict[str, Any]]]) -> str:
    """
    Create a word cloud visualization of keyword extraction results.
    
    Args:
        keyword_results: A list of lists of keyword dictionaries.
        
    Returns:
        A base64-encoded image of the word cloud.
    """
    if not keyword_results or all(not keywords for keywords in keyword_results):
        # Create an empty word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate("No keywords available")
    else:
        # Flatten the list of lists and create a frequency dictionary
        word_freq = {}
        for doc_keywords in keyword_results:
            for keyword in doc_keywords:
                word = keyword['text']
                score = keyword.get('score', 1.0)
                
                if word in word_freq:
                    word_freq[word] += score
                else:
                    word_freq[word] = score
        
        # Create a word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            random_state=42
        ).generate_from_frequencies(word_freq)
    
    # Convert the word cloud to an image
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    
    # Encode the image as base64
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{encoded_img}"

def create_summary_chart(summary_results: List[str], original_texts: List[str]) -> go.Figure:
    """
    Create a visualization of text summarization results.
    
    Args:
        summary_results: A list of summary texts.
        original_texts: A list of original texts.
        
    Returns:
        A Plotly figure object.
    """
    if not summary_results or not original_texts:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No summary data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate compression ratios
    compression_ratios = []
    for summary, original in zip(summary_results, original_texts):
        if original:
            ratio = len(summary) / len(original)
            compression_ratios.append(ratio)
    
    # Create a histogram of compression ratios
    fig = go.Figure(data=[go.Histogram(
        x=compression_ratios,
        nbinsx=20,
        marker=dict(color='rgba(50, 171, 96, 0.7)')
    )])
    
    # Update layout
    fig.update_layout(
        title="Summary Compression Ratios",
        xaxis_title="Compression Ratio (summary length / original length)",
        yaxis_title="Count",
        height=500
    )
    
    return fig

def create_dashboard(results: Dict[str, Any]) -> List[go.Figure]:
    """
    Create a dashboard with multiple visualizations.
    
    Args:
        results: A dictionary containing analysis results.
        
    Returns:
        A list of Plotly figure objects.
    """
    figures = []
    
    # Add sentiment analysis visualization
    if 'sentiment' in results:
        figures.append(create_sentiment_chart(results['sentiment']))
    
    # Add named entity recognition visualization
    if 'entities' in results:
        figures.append(create_entity_chart(results['entities']))
        figures.append(create_entity_network(results['entities']))
    
    # Add topic modeling visualization
    if 'topics' in results:
        figures.append(create_topic_chart(results['topics']))
    
    # Add text classification visualization
    if 'categories' in results:
        figures.append(create_category_chart(results['categories']))
    
    # Add keyword extraction visualization
    if 'keywords' in results:
        figures.append(create_keyword_chart(results['keywords']))
    
    # Add text summarization visualization
    if 'summaries' in results and 'original_text' in results:
        figures.append(create_summary_chart(results['summaries'], results['original_text']))
    
    return figures

def generate_wordcloud(
    tokens: List[str],
    output_path: str,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    max_words: int = 200
) -> bool:
    """Generate a word cloud from tokens."""
    if not WORDCLOUD_AVAILABLE:
        logger.warning("WordCloud is not available. Word cloud generation will be skipped.")
        return False
    
    if not tokens:
        logger.warning("No tokens provided for word cloud generation.")
        return False
    
    try:
        # Create a frequency dictionary
        text = ' '.join(tokens)
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words
        ).generate(text)
        
        # Save the word cloud
        wordcloud.to_file(output_path)
        
        return True
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return False

def generate_sentiment_chart(
    sentiment_data: Dict[str, Any],
    output_path: str,
    width: int = 8,
    height: int = 6,
    dpi: int = 100
) -> bool:
    """Generate a sentiment chart."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib is not available. Sentiment chart generation will be skipped.")
        return False
    
    if not sentiment_data:
        logger.warning("No sentiment data provided for chart generation.")
        return False
    
    try:
        # Create a figure
        plt.figure(figsize=(width, height), dpi=dpi)
        
        # Extract sentiment scores
        scores = {
            'Positive': sentiment_data.get('positive', 0),
            'Negative': sentiment_data.get('negative', 0),
            'Neutral': sentiment_data.get('neutral', 0)
        }
        
        # Create a bar chart
        plt.bar(
            scores.keys(),
            scores.values(),
            color=[COLOR_SCHEMES['sentiment'][k.lower()] for k in scores.keys()]
        )
        
        # Add labels and title
        plt.xlabel('Sentiment')
        plt.ylabel('Score')
        plt.title('Sentiment Analysis')
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the chart
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Error generating sentiment chart: {str(e)}")
        return False

def generate_entity_chart(
    entities: List[Dict[str, Any]],
    output_path: str,
    width: int = 10,
    height: int = 6,
    dpi: int = 100,
    max_entities: int = 10
) -> bool:
    """Generate an entity chart."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib is not available. Entity chart generation will be skipped.")
        return False
    
    if not entities:
        logger.warning("No entities provided for chart generation.")
        return False
    
    try:
        # Create a figure
        plt.figure(figsize=(width, height), dpi=dpi)
        
        # Count entity types
        entity_counts = Counter(entity['type'] for entity in entities)
        
        # Get the top N entity types
        top_entities = entity_counts.most_common(max_entities)
        types, counts = zip(*top_entities)
        
        # Create colors list
        colors = [COLOR_SCHEMES['entity_types'].get(t, '#95a5a6') for t in types]
        
        # Create a bar chart
        plt.bar(types, counts, color=colors)
        
        # Add labels and title
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.title('Named Entity Recognition')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Error generating entity chart: {str(e)}")
        return False

def generate_keyword_chart(
    keywords: List[Dict[str, Any]],
    output_path: str,
    width: int = 10,
    height: int = 6,
    dpi: int = 100,
    max_keywords: int = 10
) -> bool:
    """Generate a keyword chart."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib is not available. Keyword chart generation will be skipped.")
        return False
    
    if not keywords:
        logger.warning("No keywords provided for chart generation.")
        return False
    
    try:
        # Create a figure
        plt.figure(figsize=(width, height), dpi=dpi)
        
        # Sort keywords by score and get top N
        sorted_keywords = sorted(keywords, key=lambda x: x.get('score', 0), reverse=True)[:max_keywords]
        
        # Extract labels and values
        labels = [kw.get('text', '') for kw in sorted_keywords]
        scores = [kw.get('score', 0) for kw in sorted_keywords]
        
        # Create a horizontal bar chart
        plt.barh(labels, scores, color='#3498db')
        
        # Add labels and title
        plt.xlabel('Score')
        plt.ylabel('Keyword')
        plt.title('Keyword Extraction')
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Error generating keyword chart: {str(e)}")
        return False 