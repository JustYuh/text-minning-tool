/**
 * Text Mining Tool - Main JavaScript
 * This file contains client-side functionality for the Text Mining Tool web interface.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const processForm = document.getElementById('process-form');
    if (processForm) {
        processForm.addEventListener('submit', function(event) {
            const textInput = document.getElementById('text');
            const fileInput = document.getElementById('file');
            
            // Check if either text or file is provided
            if (!textInput.value.trim() && (!fileInput.files || fileInput.files.length === 0)) {
                event.preventDefault();
                showError('Please provide either text input or upload a file.');
                return false;
            }
            
            // Show loading indicator
            showLoading();
            return true;
        });
    }
    
    // File input handling
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileNameDisplay = document.getElementById('file-name');
            if (fileNameDisplay) {
                if (fileInput.files && fileInput.files.length > 0) {
                    fileNameDisplay.textContent = `Selected file: ${fileInput.files[0].name}`;
                    fileNameDisplay.style.display = 'block';
                } else {
                    fileNameDisplay.style.display = 'none';
                }
            }
        });
    }
    
    // API form handling
    const apiForm = document.getElementById('api-form');
    if (apiForm) {
        apiForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const textInput = document.getElementById('api-text');
            const analysesInputs = document.querySelectorAll('input[name="api-analyses"]:checked');
            
            if (!textInput.value.trim()) {
                showError('Please provide text input for API processing.');
                return false;
            }
            
            // Collect selected analyses
            const analyses = [];
            analysesInputs.forEach(function(input) {
                analyses.push(input.value);
            });
            
            // Prepare request data
            const requestData = {
                text: textInput.value,
                analyses: analyses
            };
            
            // Show loading indicator
            showLoading();
            
            // Send AJAX request
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                hideLoading();
                displayApiResults(data);
            })
            .catch(error => {
                hideLoading();
                showError('Error processing text: ' + error.message);
            });
        });
    }
    
    // Toggle sections
    const toggleButtons = document.querySelectorAll('.toggle-button');
    if (toggleButtons) {
        toggleButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                const targetId = this.getAttribute('data-target');
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    if (targetElement.style.display === 'none') {
                        targetElement.style.display = 'block';
                        this.textContent = 'Hide';
                    } else {
                        targetElement.style.display = 'none';
                        this.textContent = 'Show';
                    }
                }
            });
        });
    }
    
    // Download results
    const downloadButtons = document.querySelectorAll('.download-button');
    if (downloadButtons) {
        downloadButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                const format = this.getAttribute('data-format');
                const resultData = document.getElementById('result-data');
                
                if (resultData && resultData.textContent) {
                    const data = JSON.parse(resultData.textContent);
                    downloadResults(data, format);
                }
            });
        });
    }
});

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(function() {
            errorElement.style.display = 'none';
        }, 5000);
    } else {
        alert(message);
    }
}

/**
 * Show loading indicator
 */
function showLoading() {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
    
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
    }
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
    
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = false;
    }
}

/**
 * Display API results
 * @param {Object} data - Result data from API
 */
function displayApiResults(data) {
    const resultsContainer = document.getElementById('api-results');
    if (!resultsContainer) return;
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    // Create results HTML
    let html = '<h3>API Results</h3>';
    
    // Original text
    if (data.processed && data.processed.original_text) {
        html += `
            <div class="result-card">
                <div class="result-card-header">Original Text</div>
                <div class="result-card-body">
                    <p>${data.processed.original_text}</p>
                </div>
            </div>
        `;
    }
    
    // Tokens
    if (data.processed && data.processed.tokens) {
        html += `
            <div class="result-card">
                <div class="result-card-header">Tokens (${data.processed.tokens.length})</div>
                <div class="result-card-body">
                    <div class="token-list">
                        ${data.processed.tokens.map(token => `<span class="token">${token}</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Sentences
    if (data.processed && data.processed.sentences) {
        html += `
            <div class="result-card">
                <div class="result-card-header">Sentences (${data.processed.sentences.length})</div>
                <div class="result-card-body">
                    <div class="sentence-list">
                        ${data.processed.sentences.map(sentence => `<div class="sentence">${sentence}</div>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Sentiment
    if (data.sentiment) {
        const sentimentClass = data.sentiment.label === 'positive' ? 'sentiment-positive' : 
                              data.sentiment.label === 'negative' ? 'sentiment-negative' : 
                              'sentiment-neutral';
        
        html += `
            <div class="result-card">
                <div class="result-card-header">Sentiment Analysis</div>
                <div class="result-card-body">
                    <div class="sentiment ${sentimentClass}">
                        <div class="result-title">${data.sentiment.label.charAt(0).toUpperCase() + data.sentiment.label.slice(1)} Sentiment (${data.sentiment.score.toFixed(2)})</div>
                        <p>The text expresses a ${data.sentiment.label} sentiment.</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Entities
    if (data.entities && data.entities.length > 0) {
        html += `
            <div class="result-card">
                <div class="result-card-header">Named Entities</div>
                <div class="result-card-body">
                    <div class="entity-list">
                        ${data.entities.map(entity => `<span class="entity entity-${entity.type.toLowerCase()}">${entity.text} (${entity.type})</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Keywords
    if (data.keywords && data.keywords.length > 0) {
        html += `
            <div class="result-card">
                <div class="result-card-header">Keywords</div>
                <div class="result-card-body">
                    <div class="keyword-list">
                        ${data.keywords.map(keyword => `<span class="keyword">${keyword.text} (${keyword.score.toFixed(2)})</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Raw JSON
    html += `
        <div class="result-card">
            <div class="result-card-header">Raw JSON</div>
            <div class="result-card-body">
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </div>
        </div>
    `;
    
    // Add download buttons
    html += `
        <div class="download-buttons">
            <button class="btn btn-primary download-button" data-format="json">Download JSON</button>
            <button class="btn btn-primary download-button" data-format="csv">Download CSV</button>
            <button class="btn btn-primary download-button" data-format="txt">Download Text</button>
        </div>
    `;
    
    // Set HTML
    resultsContainer.innerHTML = html;
    resultsContainer.style.display = 'block';
    
    // Add event listeners to new download buttons
    const newDownloadButtons = resultsContainer.querySelectorAll('.download-button');
    if (newDownloadButtons) {
        newDownloadButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                const format = this.getAttribute('data-format');
                downloadResults(data, format);
            });
        });
    }
    
    // Store result data for later use
    const resultDataElement = document.createElement('div');
    resultDataElement.id = 'result-data';
    resultDataElement.style.display = 'none';
    resultDataElement.textContent = JSON.stringify(data);
    resultsContainer.appendChild(resultDataElement);
}

/**
 * Download results in specified format
 * @param {Object} data - Result data
 * @param {string} format - Format to download (json, csv, txt)
 */
function downloadResults(data, format) {
    let content = '';
    let filename = 'text_mining_results';
    let contentType = '';
    
    if (format === 'json') {
        content = JSON.stringify(data, null, 2);
        filename += '.json';
        contentType = 'application/json';
    } else if (format === 'csv') {
        // Create CSV content
        const headers = ['original_text', 'tokens', 'sentences'];
        content = headers.join(',') + '\n';
        
        if (data.processed) {
            const row = [
                `"${data.processed.original_text.replace(/"/g, '""')}"`,
                `"${data.processed.tokens.join(' ').replace(/"/g, '""')}"`,
                `"${data.processed.sentences.join(' ').replace(/"/g, '""')}"`
            ];
            content += row.join(',');
        }
        
        filename += '.csv';
        contentType = 'text/csv';
    } else if (format === 'txt') {
        // Create plain text content
        content = 'Text Mining Results\n\n';
        
        if (data.processed) {
            content += 'Original Text:\n' + data.processed.original_text + '\n\n';
            content += 'Tokens:\n' + data.processed.tokens.join(', ') + '\n\n';
            content += 'Sentences:\n' + data.processed.sentences.join('\n') + '\n\n';
        }
        
        if (data.sentiment) {
            content += 'Sentiment:\n';
            content += `Label: ${data.sentiment.label}\n`;
            content += `Score: ${data.sentiment.score.toFixed(2)}\n\n`;
        }
        
        if (data.entities && data.entities.length > 0) {
            content += 'Named Entities:\n';
            data.entities.forEach(entity => {
                content += `${entity.text} (${entity.type})\n`;
            });
            content += '\n';
        }
        
        if (data.keywords && data.keywords.length > 0) {
            content += 'Keywords:\n';
            data.keywords.forEach(keyword => {
                content += `${keyword.text} (${keyword.score.toFixed(2)})\n`;
            });
            content += '\n';
        }
        
        filename += '.txt';
        contentType = 'text/plain';
    }
    
    // Create download link
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    
    // Clean up
    URL.revokeObjectURL(url);
} 