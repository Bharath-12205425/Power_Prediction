from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app
from models import PowerNode, PowerEdge, NodeStatus, Prediction, MLModel, data_store
from data_processor import DataProcessor
from ml_pipeline import MLPipeline
from action_logger import action_logger
import json
import logging
import traceback
from datetime import datetime, timedelta
import os

# Initialize components
data_processor = DataProcessor()
ml_pipeline = MLPipeline()

# Initialize data flag
_data_initialized = False

@app.before_request
def initialize_data():
    """Initialize data and models on first request"""
    global _data_initialized
    if not _data_initialized:
        try:
            app.logger.info("Initializing data on startup...")
            
            # Try to load data
            if data_processor.load_data():
                app.logger.info("Data loaded successfully")
                
                # Initialize node statuses if they don't exist
                if len(data_store.get_all_node_statuses()) == 0:
                    data_processor.initialize_node_statuses()
                    app.logger.info("Node statuses initialized")
                    
                # Initialize ML models if they don't exist
                if len(data_store.get_ml_models(active_only=False)) == 0:
                    ml_pipeline.initialize_models()
                    app.logger.info("ML models initialized")
            else:
                app.logger.warning("Could not load data - running without historical data")
                
            _data_initialized = True
        except Exception as e:
            app.logger.error(f"Error during initialization: {e}")
            app.logger.error(traceback.format_exc())

@app.route('/')
def index():
    """Home page with overview"""
    try:
        action_logger.log_route_access('/', 'GET')
        nodes = data_store.get_nodes()
        edges = data_store.get_edges()
        statuses = data_store.get_all_node_statuses()
        predictions = data_store.get_predictions()

        total_nodes = len(nodes)
        total_edges = len(edges)
        operational_nodes = len([s for s in statuses if s.status == 'operational'])
        recent_predictions = len([p for p in predictions
                                if p.created_at >= datetime.utcnow() - timedelta(days=1)])

        stats = {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'operational_nodes': operational_nodes,
            'recent_predictions': recent_predictions
        }

        return render_template('index.html', stats=stats)
    except Exception as e:
        app.logger.error(f"Error in index route: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("INDEX_PAGE_LOAD", str(e))
        return render_template('index.html', stats={})

@app.route('/dashboard')
def dashboard():
    """Dashboard with real-time monitoring"""
    try:
        action_logger.log_route_access('/dashboard', 'GET')
        stats = data_processor.get_dashboard_stats()
        nodes = data_store.get_nodes()
        nodes_data = []

        for node in nodes[:10]:  # Limit to 10 for display
            status = data_store.get_node_status(node.node_id)
            node_data = node.to_dict()

            if status:
                status_data = status.to_dict()
                node_data.update({
                    'status': status_data['status'],
                    'current_risk_score': status_data['current_risk_score'],
                    'last_maintenance_days': (
                        (datetime.utcnow() - status.last_maintenance_date).days
                        if status.last_maintenance_date else None
                    )
                })
            else:
                node_data.update({
                    'status': 'unknown',
                    'current_risk_score': 0,
                    'last_maintenance_days': None
                })

            nodes_data.append(node_data)

        return render_template('dashboard.html', stats=stats, nodes=nodes_data)
    except Exception as e:
        app.logger.error(f"Error in dashboard route: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("DASHBOARD_PAGE_LOAD", str(e))
        return render_template('dashboard.html', stats={}, nodes=[])

@app.route('/predictions')
def predictions():
    """Predictions interface"""
    try:
        action_logger.log_route_access('/predictions', 'GET')
        nodes = data_store.get_nodes()
        recent_predictions = data_store.get_predictions(limit=20)
        models = data_store.get_ml_models(active_only=True)

        # Fix: ensure node dropdown shows all nodes and model dropdown works
        node_options = [{'id': n.node_id, 'name': f"{n.node_id} ({n.building_type})"} for n in nodes]
        model_options = [{'name': m.name, 'version': m.version} for m in models]

        return render_template('predictions.html',
                             nodes=node_options,
                             recent_predictions=recent_predictions,
                             models=model_options)
    except Exception as e:
        app.logger.error(f"Error in predictions route: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("PREDICTIONS_PAGE_LOAD", str(e))
        return render_template('predictions.html', nodes=[], recent_predictions=[], models=[])

@app.route('/network')
def network():
    """Network visualization page"""
    try:
        action_logger.log_route_access('/network', 'GET')
        network_data = data_processor.get_network_data()
        return render_template('network.html', network_data=network_data)
    except Exception as e:
        app.logger.error(f"Error in network route: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("NETWORK_PAGE_LOAD", str(e))
        return render_template('network.html', network_data={})

# API Routes

@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    try:
        stats = data_processor.get_dashboard_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        app.logger.error(f"Error getting dashboard stats: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/nodes')
def api_nodes():
    try:
        nodes = data_store.get_nodes()
        nodes_data = []
        
        for node in nodes:
            status = data_store.get_node_status(node.node_id)
            node_data = node.to_dict()
            
            if status:
                status_data = status.to_dict()
                node_data.update({
                    'status': status_data['status'],
                    'current_risk_score': status_data['current_risk_score'],
                    'last_maintenance_days': (
                        (datetime.utcnow() - status.last_maintenance_date).days 
                        if status.last_maintenance_date else None
                    )
                })
            else:
                node_data.update({
                    'status': 'unknown',
                    'current_risk_score': 0,
                    'last_maintenance_days': None
                })
            
            nodes_data.append(node_data)
        
        return jsonify({'success': True, 'nodes': nodes_data})
    except Exception as e:
        app.logger.error(f"Error getting nodes: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/network-data')
def api_network_data():
    try:
        network_data = data_processor.get_network_data()
        return jsonify({'success': True, 'data': network_data})
    except Exception as e:
        app.logger.error(f"Error getting network data: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        if not data or 'node_id' not in data:
            return jsonify({'success': False, 'error': 'Node ID is required'}), 400

        node_id = data['node_id']
        model_name = data.get('model_name', None)
        custom_features = data.get('features', None)

        action_logger.log_ml_operation("PREDICTION_REQUEST", model_name, node_id, f"Features: {custom_features}")

        result = ml_pipeline.predict_failure(node_id, model_name, custom_features)

        # Fetch current model performance for all models
        models = data_store.get_ml_models(active_only=True)
        performance = {}
        for model in models:
            performance[model.name] = {
                'accuracy': model.accuracy,
                'precision': model.precision,
                'recall': model.recall,
                'f1_score': model.f1_score,
                'roc_auc': model.roc_auc,
                'version': model.version,
                'algorithm': model.algorithm
            }

        if result:
            prediction = Prediction(
                node_id=node_id,
                probability=result['probability'],
                risk_level=result['risk_level'],
                model_name=result['model_name'],
                model_version=result.get('model_version', '1.0'),
                features_used=json.dumps(result.get('features_used', {}))
            )
            data_store.add_prediction(prediction)

            action_logger.log_ml_operation("PREDICTION_SUCCESS", model_name, node_id, f"Probability: {result['probability']}, Risk: {result['risk_level']}")

            result['success'] = True
            result['timestamp'] = datetime.utcnow().isoformat()
            result['prediction'] = result['probability'] > 0.5
            result['model_performance'] = performance  # <-- added this line

            return jsonify(result)
        else:
            action_logger.log_error("PREDICTION_FAILED", f"Node: {node_id}, Model: {model_name}")
            return jsonify({'success': False, 'error': 'Prediction failed', 'model_performance': performance}), 500

    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("PREDICTION_ERROR", f"Node: {node_id}, Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-performance')
def api_model_performance():
    try:
        models = data_store.get_ml_models(active_only=True)
        performance = {}
        
        for model in models:
            performance[model.name] = {
                'accuracy': model.accuracy,
                'precision': model.precision,
                'recall': model.recall,
                'f1_score': model.f1_score,
                'roc_auc': model.roc_auc,
                'version': model.version,
                'algorithm': model.algorithm
            }
        
        return jsonify({'success': True, 'performance': performance})
    except Exception as e:
        app.logger.error(f"Error getting model performance: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/load-sample-data', methods=['POST'])
def api_load_sample_data():
    try:
        action_logger.log_data_operation("LOAD_SAMPLE_DATA", "sample_data")
        success = data_processor.load_sample_data()
        if success:
            action_logger.log_data_operation("LOAD_SAMPLE_DATA_SUCCESS", "sample_data", result="Data loaded successfully")
            return jsonify({'success': True, 'message': 'Sample data loaded successfully'})
        else:
            action_logger.log_error("LOAD_SAMPLE_DATA_FAILED", "Failed to load sample data")
            return jsonify({'success': False, 'error': 'Failed to load sample data'}), 500
    except Exception as e:
        app.logger.error(f"Error loading sample data: {e}")
        app.logger.error(traceback.format_exc())
        action_logger.log_error("LOAD_SAMPLE_DATA_ERROR", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers

@app.errorhandler(404)
def not_found_error(error):
    return "404 Not Found: The requested URL was not found on the server.", 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal Server Error: {error}")
    return "500 Internal Server Error: Something went wrong on the server.", 500

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return "", 204

# Template filters

@app.template_filter('tojsonfilter')
def to_json_filter(obj):
    """Convert object to JSON for template use"""
    return json.dumps(obj)
