// Predictions functionality
class PredictionsManager {
    constructor() {
        this.models = [];
        this.nodes = [];
        this.recentPredictions = [];
        this.historyChart = null;
    }

    init() {
        this.initializeForm();
        this.initializeHistoryChart();
        this.loadAvailableModels();
        this.loadAvailableNodes(); // <-- added for node dropdown
        this.loadRecentPredictions();
    }

    initializeForm() {
        const form = document.getElementById('predictionForm');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.makePrediction();
        });
    }

    initializeHistoryChart() {
        const ctx = document.getElementById('historyChart');
        if (!ctx) return;

        this.historyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Failure Probability',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#fff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/model-performance');
            const data = await response.json();
            
            if (data.success && data.performance) {
                const modelSelect = document.getElementById('modelSelect');
                modelSelect.innerHTML = '<option value="">Use Best Model</option>';
                
                Object.keys(data.performance).forEach(modelName => {
                    const option = document.createElement('option');
                    option.value = modelName;
                    option.textContent = modelName;
                    modelSelect.appendChild(option);
                });
                
                this.models = Object.keys(data.performance);
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async loadAvailableNodes() {
        try {
            const response = await fetch('/api/nodes');
            const data = await response.json();

            if (data.success && data.nodes) {
                const nodeSelect = document.getElementById('nodeSelect');
                nodeSelect.innerHTML = '<option value="">Choose a node...</option>';

                data.nodes.forEach(node => {
                    const option = document.createElement('option');
                    option.value = node.node_id;
                    option.textContent = node.node_name || node.node_id;
                    nodeSelect.appendChild(option);
                });

                this.nodes = data.nodes;
            }
        } catch (error) {
            console.error('Error loading nodes:', error);
        }
    }

    async makePrediction() {
        const nodeId = document.getElementById('nodeSelect').value;
        const modelName = document.getElementById('modelSelect').value;
        
        if (!nodeId) {
            showNotification('Please select a node', 'warning');
            return;
        }

        const predictBtn = document.getElementById('predictBtn');
        const originalText = predictBtn.innerHTML;
        
        // Show loading state
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Predicting...';
        predictBtn.disabled = true;

        try {
            // Collect custom features if provided
            const features = this.collectCustomFeatures();
            
            const requestBody = {
                node_id: nodeId,
                features: Object.keys(features).length > 0 ? features : null
            };
            
            if (modelName) {
                requestBody.model_name = modelName;
            }

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (data.success) {
                this.displayPredictionResult(data);
                this.storePrediction(data);
                showNotification('Prediction completed successfully', 'success');
                this.loadRecentPredictions(); // Refresh recent predictions
            } else {
                showNotification(`Prediction failed: ${data.error}`, 'danger');
                this.displayPredictionError(data.error);
            }
        } catch (error) {
            showNotification(`Error making prediction: ${error.message}`, 'danger');
            this.displayPredictionError(error.message);
        } finally {
            predictBtn.innerHTML = originalText;
            predictBtn.disabled = false;
        }
    }

    collectCustomFeatures() {
        const features = {};
        
        const featureInputs = {
            'daily_load_kwh': 'dailyLoad',
            'temp_c': 'temperature',
            'weather_severity': 'weatherSeverity',
            'vibration_g': 'vibration',
            'last_maintenance_days': 'maintenanceDays',
            'error_code': 'errorCode'
        };

        Object.entries(featureInputs).forEach(([featureName, inputId]) => {
            const input = document.getElementById(inputId);
            if (input && input.value) {
                const value = input.type === 'number' ? parseFloat(input.value) : input.value;
                features[featureName] = value;
            }
        });

        return features;
    }

    displayPredictionResult(data) {
        const container = document.getElementById('predictionResult');
        const probability = (data.probability * 100).toFixed(1);
        const prediction = data.prediction ? 'FAILURE PREDICTED' : 'NO FAILURE';
        const riskLevel = data.risk_level || 'Low';
        
        let riskColor = 'success';
        if (riskLevel === 'Critical') riskColor = 'danger';
        else if (riskLevel === 'High') riskColor = 'danger';
        else if (riskLevel === 'Medium') riskColor = 'warning';

        container.innerHTML = `
            <div class="prediction-result prediction-${riskColor}">
                <h5 class="mb-3">${data.node_id}</h5>
                <h2 class="mb-3">${probability}%</h2>
                <p class="mb-2"><strong>${prediction}</strong></p>
                <p class="mb-3">Risk Level: <span class="badge bg-${riskColor}">${riskLevel}</span></p>
                <small class="text-muted">Model: ${data.model_name || 'Default'}</small>
                <br>
                <small class="text-muted">${new Date(data.timestamp).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata'})}</small>
            </div>
        `;
    }

    displayPredictionError(error) {
        const container = document.getElementById('predictionResult');
        container.innerHTML = `
            <div class="text-center text-danger py-4">
                <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                <p>Prediction Failed</p>
                <small>${error}</small>
            </div>
        `;
    }

    async loadRecentPredictions() {
        try {
            const storedPredictions = localStorage.getItem('recentPredictions');
            let predictions = storedPredictions ? JSON.parse(storedPredictions) : [];
            
            const tableBody = document.getElementById('recentPredictionsTable');
            
            if (predictions.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-muted py-4">
                            <i class="fas fa-clock fa-2x mb-2"></i><br>
                            No recent predictions available
                        </td>
                    </tr>
                `;
                return;
            }

            const recentPredictions = predictions.slice(-10).reverse();
            
            tableBody.innerHTML = recentPredictions.map(pred => `
                <tr>
                    <td><strong>${pred.node_id}</strong></td>
                    <td>${new Date(pred.timestamp).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata'})}</td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="progress me-2" style="width: 60px; height: 8px;">
                                <div class="progress-bar ${this.getProbabilityColor(pred.probability)}" 
                                     style="width: ${(pred.probability * 100)}%"></div>
                            </div>
                            <small>${(pred.probability * 100).toFixed(1)}%</small>
                        </div>
                    </td>
                    <td>
                        <span class="badge bg-${pred.prediction ? 'danger' : 'success'}">
                            ${pred.prediction ? 'Failure' : 'No Failure'}
                        </span>
                    </td>
                    <td>
                        <span class="badge bg-${this.getRiskLevelColor(pred.risk_level)}">
                            ${pred.risk_level}
                        </span>
                    </td>
                    <td><small>${pred.model_name}</small></td>
                </tr>
            `).join('');
            
            this.updateHistoryChart(predictions);
            
        } catch (error) {
            console.error('Error loading recent predictions:', error);
        }
    }

    getProbabilityColor(probability) {
        if (probability > 0.7) return 'bg-danger';
        if (probability > 0.4) return 'bg-warning';
        return 'bg-success';
    }

    getRiskLevelColor(riskLevel) {
        switch (riskLevel) {
            case 'Critical': return 'danger';
            case 'High': return 'danger';
            case 'Medium': return 'warning';
            default: return 'success';
        }
    }

    updateHistoryChart(predictions) {
        if (!this.historyChart || predictions.length === 0) return;

        const dailyData = {};
        predictions.forEach(pred => {
            const date = new Date(pred.timestamp).toDateString();
            if (!dailyData[date]) {
                dailyData[date] = [];
            }
            dailyData[date].push(pred.probability);
        });

        const labels = Object.keys(dailyData).sort().slice(-30);
        const data = labels.map(date => {
            const probs = dailyData[date];
            return probs.reduce((sum, prob) => sum + prob, 0) / probs.length;
        });

        this.historyChart.data.labels = labels.map(date => new Date(date).toLocaleDateString());
        this.historyChart.data.datasets[0].data = data;
        this.historyChart.update();
    }

    storePrediction(prediction) {
        const storedPredictions = localStorage.getItem('recentPredictions');
        let predictions = storedPredictions ? JSON.parse(storedPredictions) : [];
        
        predictions.push(prediction);
        if (predictions.length > 100) {
            predictions = predictions.slice(-100);
        }
        
        localStorage.setItem('recentPredictions', JSON.stringify(predictions));
    }
}

// Global functions
let predictionsManager = null;

function initializePredictions() {
    predictionsManager = new PredictionsManager();
    predictionsManager.init();
}

function toggleCustomFeatures() {
    const container = document.getElementById('customFeatures');
    const button = document.getElementById('toggleFeatures');
    
    if (container.style.display === 'none') {
        container.style.display = 'block';
        button.innerHTML = '<i class="fas fa-sliders-h me-1"></i>Hide Features';
    } else {
        container.style.display = 'none';
        button.innerHTML = '<i class="fas fa-sliders-h me-1"></i>Show Features';
    }
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('customFeatures').style.display = 'none';
    document.getElementById('toggleFeatures').innerHTML = '<i class="fas fa-sliders-h me-1"></i>Show Features';
    
    document.getElementById('predictionResult').innerHTML = `
        <div class="text-center text-muted py-4">
            <i class="fas fa-chart-line fa-3x mb-3 opacity-50"></i>
            <p>Select a node and click "Predict Failure" to see results</p>
        </div>
    `;
}
