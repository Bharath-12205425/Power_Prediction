from datetime import datetime
import pandas as pd
import numpy as np

class PowerNode:
    """Model for power grid nodes - now using in-memory storage"""
    
    def __init__(self, node_id=None, building_type=None, max_capacity_kwh=None, 
                 installed_solar_kw=0, has_battery_backup=False, latitude=None, longitude=None):
        self.node_id = node_id
        self.building_type = building_type
        self.max_capacity_kwh = max_capacity_kwh
        self.installed_solar_kw = installed_solar_kw
        self.has_battery_backup = has_battery_backup
        self.latitude = latitude
        self.longitude = longitude
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            'node_id': self.node_id,
            'building_type': self.building_type,
            'max_capacity_kwh': self.max_capacity_kwh,
            'installed_solar_kw': self.installed_solar_kw,
            'has_battery_backup': self.has_battery_backup,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class PowerEdge:
    """Model for power grid connections - now using in-memory storage"""
    
    def __init__(self, source=None, target=None, capacity_mw=None, 
                 length_km=None, voltage_kv=None, line_type=None):
        self.source = source
        self.target = target
        self.capacity_mw = capacity_mw
        self.length_km = length_km
        self.voltage_kv = voltage_kv
        self.line_type = line_type
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'capacity_mw': self.capacity_mw,
            'length_km': self.length_km,
            'voltage_kv': self.voltage_kv,
            'line_type': self.line_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class NodeStatus:
    """Model for real-time node status - now using in-memory storage"""
    
    def __init__(self, node_id=None, status='operational', current_load_kwh=0,
                 temperature_c=None, humidity_percent=None, vibration_g=None,
                 error_code=None, last_maintenance_date=None, current_risk_score=0):
        self.node_id = node_id
        self.status = status
        self.current_load_kwh = current_load_kwh
        self.temperature_c = temperature_c
        self.humidity_percent = humidity_percent
        self.vibration_g = vibration_g
        self.error_code = error_code
        self.last_maintenance_date = last_maintenance_date
        self.current_risk_score = current_risk_score
        self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        return {
            'node_id': self.node_id,
            'status': self.status,
            'current_load_kwh': self.current_load_kwh,
            'temperature_c': self.temperature_c,
            'humidity_percent': self.humidity_percent,
            'vibration_g': self.vibration_g,
            'error_code': self.error_code,
            'last_maintenance_date': self.last_maintenance_date.isoformat() if self.last_maintenance_date else None,
            'current_risk_score': self.current_risk_score,
            'timestamp': self.timestamp.isoformat()
        }

class Prediction:
    """Model for failure predictions - now using in-memory storage"""
    
    def __init__(self, node_id=None, prediction_type='failure', probability=None,
                 risk_level=None, model_name=None, model_version='1.0',
                 prediction_horizon_days=7, features_used=None):
        self.node_id = node_id
        self.prediction_type = prediction_type
        self.probability = probability
        self.risk_level = risk_level
        self.model_name = model_name
        self.model_version = model_version
        self.prediction_horizon_days = prediction_horizon_days
        self.features_used = features_used
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            'node_id': self.node_id,
            'prediction_type': self.prediction_type,
            'probability': self.probability,
            'risk_level': self.risk_level,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction_horizon_days': self.prediction_horizon_days,
            'features_used': self.features_used,
            'created_at': self.created_at.isoformat()
        }

class MLModel:
    """Model for tracking ML model metadata - now using in-memory storage"""
    
    def __init__(self, name=None, version='1.0', algorithm=None, accuracy=None,
                 precision=None, recall=None, f1_score=None, roc_auc=None,
                 model_path=None, feature_names=None, is_active=True):
        self.name = name
        self.version = version
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc
        self.training_date = datetime.utcnow()
        self.is_active = is_active
        self.model_path = model_path
        self.feature_names = feature_names
    
    def to_dict(self):
        return {
            'name': self.name,
            'version': self.version,
            'algorithm': self.algorithm,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'training_date': self.training_date.isoformat(),
            'is_active': self.is_active,
            'model_path': self.model_path,
            'feature_names': self.feature_names
        }

# In-memory data storage
class DataStore:
    """In-memory data storage to replace database"""
    
    def __init__(self):
        self.nodes = {}  # node_id -> PowerNode
        self.edges = []  # List of PowerEdge objects
        self.node_statuses = {}  # node_id -> NodeStatus
        self.predictions = []  # List of Prediction objects
        self.ml_models = {}  # model_name -> MLModel
        
    def add_node(self, node):
        self.nodes[node.node_id] = node
        
    def add_edge(self, edge):
        self.edges.append(edge)
        
    def add_node_status(self, status):
        self.node_statuses[status.node_id] = status
        
    def add_prediction(self, prediction):
        self.predictions.append(prediction)
        
    def add_ml_model(self, model):
        self.ml_models[model.name] = model
        
    def get_nodes(self):
        return list(self.nodes.values())
        
    def get_node(self, node_id):
        return self.nodes.get(node_id)
        
    def get_edges(self):
        return self.edges
        
    def get_node_status(self, node_id):
        return self.node_statuses.get(node_id)
        
    def get_all_node_statuses(self):
        return list(self.node_statuses.values())
        
    def get_predictions(self, limit=None):
        predictions = sorted(self.predictions, key=lambda x: x.created_at, reverse=True)
        return predictions[:limit] if limit else predictions
        
    def get_ml_models(self, active_only=True):
        if active_only:
            return [model for model in self.ml_models.values() if model.is_active]
        return list(self.ml_models.values())
        
    def clear_all(self):
        self.nodes.clear()
        self.edges.clear()
        self.node_statuses.clear()
        self.predictions.clear()
        self.ml_models.clear()

# Global data store instance
data_store = DataStore()
