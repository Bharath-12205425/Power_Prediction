import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import logging
from datetime import datetime, timedelta
from models import PowerNode, PowerEdge, NodeStatus, data_store
import random

class DataProcessor:
    """Handles data loading, processing and analysis for the power grid system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.nodes_df = None
        self.edges_df = None
        self.timeseries_df = None
        self.graph = None
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            data_dir = 'data'
            if not os.path.exists(data_dir):
                self.logger.warning("Data directory not found, creating sample data")
                self.create_sample_data_files()
            
            nodes_file = os.path.join(data_dir, 'power_nodes.csv')
            edges_file = os.path.join(data_dir, 'power_edges.csv')
            timeseries_file = os.path.join(data_dir, 'power_timeseries.csv')

            if os.path.exists(nodes_file):
                self.nodes_df = pd.read_csv(nodes_file)
                self._load_nodes_to_memory()
            else:
                self.logger.warning("Nodes file missing")

            if os.path.exists(edges_file):
                self.edges_df = pd.read_csv(edges_file)
                self._load_edges_to_memory()
            else:
                self.logger.warning("Edges file missing")

            if os.path.exists(timeseries_file):
                self.timeseries_df = pd.read_csv(timeseries_file)
            
            self.logger.info("Data loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False

    def create_sample_data_files(self):
        """Create sample data files if they don't exist"""
        os.makedirs('data', exist_ok=True)
        np.random.seed(42)
        n_nodes = 50
        building_types = ['residential', 'commercial', 'industrial', 'hospital', 'school']
        nodes_data = {
            'node_id': [f'NODE_{i:03d}' for i in range(1, n_nodes + 1)],
            'building_type': np.random.choice(building_types, n_nodes),
            'max_capacity_kwh': np.random.normal(3000, 1000, n_nodes).clip(1000, 8000),
            'installed_solar_kw': np.random.exponential(100, n_nodes).clip(0, 500),
            'has_battery_backup': np.random.choice([True, False], n_nodes, p=[0.3, 0.7]),
            'latitude': np.random.uniform(40.0, 41.0, n_nodes),
            'longitude': np.random.uniform(-74.5, -73.5, n_nodes)
        }
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv('data/power_nodes.csv', index=False)

        edges_data = []
        for i in range(n_nodes):
            n_connections = random.randint(2, 5)
            targets = random.sample(range(n_nodes), min(n_connections, n_nodes - 1))
            for target in targets:
                if target != i:
                    edges_data.append({
                        'source': f'NODE_{i+1:03d}',
                        'target': f'NODE_{target+1:03d}',
                        'capacity_mw': random.uniform(1, 10),
                        'length_km': random.uniform(0.5, 5.0),
                        'voltage_kv': random.choice([11, 33, 132, 400]),
                        'line_type': random.choice(['overhead', 'underground'])
                    })
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv('data/power_edges.csv', index=False)

        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        timeseries_data = []
        for node_id in nodes_data['node_id']:
            for date in dates:
                timeseries_data.append({
                    'node_id': node_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'daily_load_kwh': random.uniform(1000, 5000),
                    'temp_c': random.uniform(-10, 35),
                    'weather_severity': random.randint(1, 5),
                    'vibration_g': random.uniform(0, 2),
                    'failure_occurred': random.choice([0, 1]) if random.random() < 0.02 else 0
                })
        timeseries_df = pd.DataFrame(timeseries_data)
        timeseries_df.to_csv('data/power_timeseries.csv', index=False)
        self.logger.info("Sample data files created")

    def _load_nodes_to_memory(self):
        """Load nodes data to memory"""
        if self.nodes_df is None:
            return
        try:
            if len(data_store.get_nodes()) == 0:
                for _, row in self.nodes_df.iterrows():
                    # Handle boolean conversion more robustly
                    has_battery_backup = str(row['has_battery_backup']).lower() == 'true'

                    node = PowerNode(
                        node_id=row['node_id'],
                        building_type=row['building_type'],
                        max_capacity_kwh=float(row['max_capacity_kwh']),
                        installed_solar_kw=float(row['installed_solar_kw']),
                        has_battery_backup=has_battery_backup,
                        latitude=float(row.get('latitude', 0)),
                        longitude=float(row.get('longitude', 0))
                    )
                    data_store.add_node(node)
                self.logger.info(f"Loaded {len(self.nodes_df)} nodes to memory")
        except Exception as e:
            self.logger.error(f"Error loading nodes to memory: {e}")

    def _load_edges_to_memory(self):
        """Load edges data to memory"""
        if self.edges_df is None:
            return
        try:
            if len(data_store.get_edges()) == 0:
                required_cols = {'source', 'target', 'capacity_mw', 'length_km', 'voltage_kv', 'line_type'}
                missing_cols = required_cols - set(self.edges_df.columns)
                if missing_cols:
                    self.logger.error(f"Edges CSV missing columns: {missing_cols}")
                    return
                for _, row in self.edges_df.iterrows():
                    edge = PowerEdge(
                        source=row['source'],
                        target=row['target'],
                        capacity_mw=float(row['capacity_mw']),
                        length_km=float(row['length_km']),
                        voltage_kv=float(row['voltage_kv']),
                        line_type=row['line_type']
                    )
                    data_store.add_edge(edge)
                self.logger.info(f"Loaded {len(self.edges_df)} edges to memory")
        except Exception as e:
            self.logger.error(f"Error loading edges to memory: {e}")

    def initialize_node_statuses(self):
        """Initialize node status data"""
        try:
            nodes = data_store.get_nodes()
            statuses = ['operational', 'maintenance', 'failed']
            status_weights = [0.8, 0.15, 0.05]
            
            for node in nodes:
                status = NodeStatus(
                    node_id=node.node_id,
                    status=np.random.choice(statuses, p=status_weights),
                    current_load_kwh=random.uniform(500, node.max_capacity_kwh * 0.8),
                    temperature_c=random.uniform(15, 35),
                    humidity_percent=random.uniform(30, 70),
                    vibration_g=random.uniform(0, 1.5),
                    error_code=random.choice([None, 'E001', 'E002', 'W001']) if random.random() < 0.1 else None,
                    last_maintenance_date=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
                    current_risk_score=random.uniform(0, 1)
                )
                data_store.add_node_status(status)
            
            self.logger.info(f"Initialized status for {len(nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Error initializing node statuses: {e}")

    def get_dashboard_stats(self):
        """Get statistics for dashboard"""
        try:
            nodes = data_store.get_nodes()
            statuses = data_store.get_all_node_statuses()
            predictions = data_store.get_predictions()
            
            total_nodes = len(nodes)
            operational_nodes = len([s for s in statuses if s.status == 'operational'])
            maintenance_nodes = len([s for s in statuses if s.status == 'maintenance'])
            failed_nodes = len([s for s in statuses if s.status == 'failed'])
            
            high_risk_nodes = len([s for s in statuses if s.current_risk_score > 0.7])
            medium_risk_nodes = len([s for s in statuses if 0.4 < s.current_risk_score <= 0.7])
            low_risk_nodes = len([s for s in statuses if s.current_risk_score <= 0.4])
            
            recent_predictions = len([p for p in predictions 
                                     if p.created_at >= datetime.utcnow() - timedelta(days=1)])
            
            return {
                'total_nodes': total_nodes,
                'operational_nodes': operational_nodes,
                'maintenance_nodes': maintenance_nodes,
                'failed_nodes': failed_nodes,
                'high_risk_nodes': high_risk_nodes,
                'medium_risk_nodes': medium_risk_nodes,
                'low_risk_nodes': low_risk_nodes,
                'recent_predictions': recent_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard stats: {e}")
            return {}

    def get_network_data(self):
        """Get network data for visualization"""
        try:
            nodes = data_store.get_nodes()
            edges = data_store.get_edges()
            
            nodes_data = []
            for node in nodes:
                status = data_store.get_node_status(node.node_id)
                
                node_data = {
                    'id': node.node_id,
                    'building_type': node.building_type,
                    'max_capacity_kwh': node.max_capacity_kwh,
                    'installed_solar_kw': node.installed_solar_kw,
                    'has_battery_backup': node.has_battery_backup,
                    'latitude': getattr(node, 'latitude', 0),
                    'longitude': getattr(node, 'longitude', 0)
                }
                
                if status:
                    node_data.update({
                        'status': status.status,
                        'current_risk_score': status.current_risk_score,
                        'current_load_kwh': status.current_load_kwh
                    })
                else:
                    node_data.update({
                        'status': 'unknown',
                        'current_risk_score': 0,
                        'current_load_kwh': 0
                    })
                
                nodes_data.append(node_data)
            
            edges_data = [edge.to_dict() for edge in edges]
            
            if self.graph is None:
                self._build_networkx_graph()
            
            stats = {
                'node_count': len(nodes_data),
                'edge_count': len(edges_data),
                'density': len(edges_data) / max(1, len(nodes_data) * (len(nodes_data) - 1) / 2) if len(nodes_data) > 1 else 0
            }
            
            if self.graph and len(nodes_data) > 0:
                try:
                    degree_centrality = nx.degree_centrality(self.graph)
                    betweenness_centrality = nx.betweenness_centrality(self.graph)
                    
                    for node in nodes_data:
                        node_id = node['id']
                        node['degree_centrality'] = degree_centrality.get(node_id, 0)
                        node['betweenness_centrality'] = betweenness_centrality.get(node_id, 0)
                        node['degree'] = self.graph.degree(node_id) if node_id in self.graph else 0
                except Exception as e:
                    self.logger.warning(f"Could not calculate centrality measures: {e}")
            
            return {
                'nodes': nodes_data,
                'edges': edges_data,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting network data: {e}")
            return {'nodes': [], 'edges': [], 'stats': {'node_count': 0, 'edge_count': 0, 'density': 0}}

    def _build_networkx_graph(self):
        """Build NetworkX graph for analysis"""
        try:
            self.graph = nx.Graph()
            
            # Add nodes
            nodes = data_store.get_nodes()
            for node in nodes:
                self.graph.add_node(node.node_id, **node.to_dict())
            
            # Add edges
            edges = data_store.get_edges()
            for edge in edges:
                self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            self.logger.info(f"Built NetworkX graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Error building NetworkX graph: {e}")
            self.graph = None
