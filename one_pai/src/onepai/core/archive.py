"""ONEPAI Dashboard UI Components

Componenti per l'interfaccia web della dashboard ONEPAI.
Fornisce visualizzazioni interattive per l'analisi delle ombre cognitive.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.shadow_mapper import ShadowMapper
from ..analysis.silence_metrics import calculate_silence_entropy
from ..memory.unsaid_vault import UnsaidVault


class DashboardUI:
    """Componente principale per l'interfaccia della dashboard."""
    
    def __init__(self):
        self.shadow_mapper = ShadowMapper()
        self.vault = UnsaidVault()
        self._cache = {}
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Raccoglie tutti i dati necessari per la dashboard."""
        return {
            'overview': self.get_overview_stats(),
            'shadows': self.get_shadow_analysis(),
            'silences': self.get_silence_metrics(),
            'treasures': self.get_treasure_summary(),
            'recent_activity': self.get_recent_activity(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_overview_stats(self) -> Dict[str, Any]:
        """Statistiche generali per la panoramica."""
        return {
            'total_shadows': 156,
            'active_silences': 23,
            'treasure_count': 42,
            'cognitive_voids': 8,
            'analysis_sessions': 15,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_shadow_analysis(self) -> Dict[str, Any]:
        """Dati per l'analisi delle ombre cognitive."""
        # Simula dati di analisi delle ombre
        shadow_data = {
            'distribution': {
                'layer_1': 25,
                'layer_2': 18,
                'layer_3': 32,
                'layer_4': 15,
                'layer_5': 22,
                'layer_6': 44
            },
            'intensity_levels': {
                'low': 45,
                'medium': 78,
                'high': 33
            },
            'temporal_patterns': [
                {'time': '00:00', 'count': 12},
                {'time': '04:00', 'count': 8},
                {'time': '08:00', 'count': 25},
                {'time': '12:00', 'count': 35},
                {'time': '16:00', 'count': 28},
                {'time': '20:00', 'count': 18}
            ],
            'top_shadow_types': [
                {'type': 'Cognitive Void', 'count': 45, 'percentage': 28.8},
                {'type': 'Silent Neuron', 'count': 38, 'percentage': 24.4},
                {'type': 'Aborted Thought', 'count': 32, 'percentage': 20.5},
                {'type': 'Hidden Circuit', 'count': 25, 'percentage': 16.0},
                {'type': 'Suppressed Pattern', 'count': 16, 'percentage': 10.3}
            ]
        }
        
        return shadow_data
    
    def get_silence_metrics(self) -> Dict[str, Any]:
        """Metriche sui pattern di silenzio."""
        # Simula calcoli di metriche del silenzio
        silence_data = [0.1, 0.2, 0.0, 0.3, 0.15, 0.05]
        entropy = calculate_silence_entropy(silence_data)
        
        return {
            'entropy': round(entropy, 4),
            'silence_ratio': 0.234,
            'void_depth': 0.156,
            'suppression_patterns': {
                'sequential': 12,
                'random': 8,
                'clustered': 15,
                'periodic': 6
            },
            'silence_timeline': [
                {'timestamp': '2024-01-01T10:00:00', 'value': 0.12},
                {'timestamp': '2024-01-01T11:00:00', 'value': 0.18},
                {'timestamp': '2024-01-01T12:00:00', 'value': 0.25},
                {'timestamp': '2024-01-01T13:00:00', 'value': 0.15},
                {'timestamp': '2024-01-01T14:00:00', 'value': 0.22}
            ]
        }
    
    def get_treasure_summary(self) -> Dict[str, Any]:
        """Riassunto dei tesori nell'archivio."""
        return {
            'total_treasures': 42,
            'categories': {
                'cognitive_patterns': 15,
                'neural_insights': 12,
                'behavioral_traces': 8,
                'decision_fragments': 7
            },
            'recent_discoveries': [
                {
                    'id': 'treasure_001',
                    'name': 'Hidden Decision Pattern',
                    'category': 'cognitive_patterns',
                    'discovered_at': '2024-01-15T14:30:00',
                    'significance': 0.87
                },
                {
                    'id': 'treasure_002', 
                    'name': 'Silent Neuron Cluster',
                    'category': 'neural_insights',
                    'discovered_at': '2024-01-15T12:15:00',
                    'significance': 0.72
                },
                {
                    'id': 'treasure_003',
                    'name': 'Suppressed Logic Path',
                    'category': 'behavioral_traces',
                    'discovered_at': '2024-01-15T09:45:00',
                    'significance': 0.65
                }
            ],
            'storage_stats': {
                'encrypted_size': '2.4 MB',
                'compression_ratio': 0.23,
                'integrity_score': 0.99
            }
        }
    
    def get_recent_activity(self) -> List[Dict[str, Any]]:
        """Attività recenti nel sistema."""
        return [
            {
                'timestamp': '2024-01-15T15:30:00',
                'type': 'analysis',
                'description': 'Shadow mapping completed for model GPT-4',
                'status': 'success',
                'details': {'shadows_found': 23, 'processing_time': '2.3s'}
            },
            {
                'timestamp': '2024-01-15T14:45:00',
                'type': 'treasure',
                'description': 'New treasure discovered: Hidden Decision Pattern',
                'status': 'success',
                'details': {'significance': 0.87, 'category': 'cognitive_patterns'}
            },
            {
                'timestamp': '2024-01-15T13:20:00',
                'type': 'silence',
                'description': 'Silence analysis completed',
                'status': 'success',
                'details': {'entropy': 0.1234, 'patterns_found': 15}
            },
            {
                'timestamp': '2024-01-15T12:10:00',
                'type': 'archive',
                'description': 'Archive backup created',
                'status': 'success',
                'details': {'size': '2.4 MB', 'treasures': 42}
            },
            {
                'timestamp': '2024-01-15T11:30:00',
                'type': 'error',
                'description': 'Failed to process model weights',
                'status': 'error',
                'details': {'error_code': 'WEIGHT_001', 'retry_count': 3}
            }
        ]
    
    def get_visualization_data(self, viz_type: str) -> Dict[str, Any]:
        """Dati specifici per diversi tipi di visualizzazione."""
        if viz_type == 'shadow_network':
            return self._get_shadow_network_data()
        elif viz_type == 'silence_heatmap':
            return self._get_silence_heatmap_data()
        elif viz_type == 'treasure_map':
            return self._get_treasure_map_data()
        elif viz_type == 'cognitive_flow':
            return self._get_cognitive_flow_data()
        else:
            return {'error': f'Unknown visualization type: {viz_type}'}
    
    def _get_shadow_network_data(self) -> Dict[str, Any]:
        """Dati per la visualizzazione della rete di ombre."""
        return {
            'nodes': [
                {'id': 'shadow_1', 'label': 'Cognitive Void', 'size': 25, 'color': '#ff6b6b'},
                {'id': 'shadow_2', 'label': 'Silent Neuron', 'size': 18, 'color': '#4ecdc4'},
                {'id': 'shadow_3', 'label': 'Hidden Circuit', 'size': 32, 'color': '#45b7d1'},
                {'id': 'shadow_4', 'label': 'Aborted Thought', 'size': 15, 'color': '#f9ca24'},
                {'id': 'shadow_5', 'label': 'Suppressed Pattern', 'size': 22, 'color': '#6c5ce7'}
            ],
            'edges': [
                {'from': 'shadow_1', 'to': 'shadow_2', 'weight': 0.8},
                {'from': 'shadow_2', 'to': 'shadow_3', 'weight': 0.6},
                {'from': 'shadow_3', 'to': 'shadow_4', 'weight': 0.4},
                {'from': 'shadow_4', 'to': 'shadow_5', 'weight': 0.7},
                {'from': 'shadow_5', 'to': 'shadow_1', 'weight': 0.3}
            ]
        }
    
    def _get_silence_heatmap_data(self) -> Dict[str, Any]:
        """Dati per la heatmap dei silenzi."""
        import random
        
        # Genera una matrice di dati per la heatmap
        layers = ['Input', 'Hidden1', 'Hidden2', 'Hidden3', 'Output']
        neurons = list(range(1, 21))  # 20 neuroni per layer
        
        data = []
        for i, layer in enumerate(layers):
            for j, neuron in enumerate(neurons):
                data.append({
                    'layer': layer,
                    'neuron': neuron,
                    'silence_level': round(random.uniform(0, 1), 3),
                    'x': i,
                    'y': j
                })
        
        return {
            'data': data,
            'layers': layers,
            'neurons': neurons,
            'color_scale': ['#ffffff', '#ff6b6b']
        }
    
    def _get_treasure_map_data(self) -> Dict[str, Any]:
        """Dati per la mappa dei tesori."""
        return {
            'treasures': [
                {'x': 120, 'y': 80, 'size': 15, 'type': 'cognitive_pattern', 'value': 0.87},
                {'x': 200, 'y': 150, 'size': 12, 'type': 'neural_insight', 'value': 0.72},
                {'x': 80, 'y': 200, 'size': 18, 'type': 'behavioral_trace', 'value': 0.91},
                {'x': 300, 'y': 100, 'size': 10, 'type': 'decision_fragment', 'value': 0.65},
                {'x': 250, 'y': 250, 'size': 20, 'type': 'cognitive_pattern', 'value': 0.94}
            ],
            'regions': [
                {'name': 'Cognitive Zone', 'bounds': [50, 50, 150, 150], 'color': '#ff6b6b33'},
                {'name': 'Neural Zone', 'bounds': [180, 120, 280, 220], 'color': '#4ecdc433'},
                {'name': 'Behavioral Zone', 'bounds': [60, 180, 160, 280], 'color': '#45b7d133'}
            ]
        }
    
    def _get_cognitive_flow_data(self) -> Dict[str, Any]:
        """Dati per il flusso cognitivo."""
        return {
            'flow_paths': [
                {
                    'id': 'path_1',
                    'points': [{'x': 50, 'y': 100}, {'x': 150, 'y': 120}, {'x': 250, 'y': 80}],
                    'intensity': 0.8,
                    'type': 'active'
                },
                {
                    'id': 'path_2', 
                    'points': [{'x': 80, 'y': 200}, {'x': 180, 'y': 180}, {'x': 280, 'y': 160}],
                    'intensity': 0.3,
                    'type': 'suppressed'
                },
                {
                    'id': 'path_3',
                    'points': [{'x': 100, 'y': 50}, {'x': 200, 'y': 70}, {'x': 300, 'y': 90}],
                    'intensity': 0.6,
                    'type': 'shadow'
                }
            ],
            'decision_points': [
                {'x': 150, 'y': 120, 'type': 'branch', 'confidence': 0.85},
                {'x': 180, 'y': 180, 'type': 'merge', 'confidence': 0.42},
                {'x': 200, 'y': 70, 'type': 'abort', 'confidence': 0.15}
            ]
        }
    
    def export_data(self, format_type: str = 'json') -> str:
        """Esporta tutti i dati della dashboard nel formato specificato."""
        data = self.get_dashboard_data()
        
        if format_type == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format_type == 'csv':
            # Implementazione semplificata per CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Scrivi header
            writer.writerow(['Type', 'Key', 'Value'])
            
            # Scrivi dati overview
            for key, value in data['overview'].items():
                writer.writerow(['overview', key, value])
                
            return output.getvalue()
        else:
            raise ValueError(f"Formato non supportato: {format_type}")
    
    def clear_cache(self):
        """Pulisce la cache dei dati."""
        self._cache.clear()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Stato di salute del sistema."""
        return {
            'status': 'healthy',
            'uptime': '2d 14h 32m',
            'memory_usage': '245 MB',
            'cpu_usage': '12%',
            'active_connections': 3,
            'last_error': None,
            'components': {
                'shadow_mapper': 'online',
                'treasure_archive': 'online',
                'silence_tracer': 'online',
                'void_analyzer': 'online'
            }
        }