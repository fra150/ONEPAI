"""Frontend utilities for processing graph data and metadata."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def add_graph_metadata(graph_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Add metadata to graph data for frontend consumption.
    
    Args:
        graph_data: The graph data dictionary
        metadata: Additional metadata to include
        
    Returns:
        Updated graph data with metadata
    """
    if 'metadata' not in graph_data:
        graph_data['metadata'] = {}
    
    graph_data['metadata'].update(metadata)
    
    # Add common metadata fields
    graph_data['metadata'].update({
        'version': '1.0',
        'format': 'circuit_tracer_graph',
        'created_by': 'circuit_tracer'
    })
    
    return graph_data


def process_token(token: str, token_id: Optional[int] = None) -> Dict[str, Any]:
    """Process a token for frontend display.
    
    Args:
        token: The token string
        token_id: Optional token ID
        
    Returns:
        Processed token data
    """
    return {
        'text': token,
        'id': token_id,
        'length': len(token),
        'is_special': token.startswith('<') and token.endswith('>'),
        'display_text': token.replace('Ġ', ' ').replace('▁', ' ')  # Handle common tokenizer prefixes
    }


def process_tokens(tokens: List[str], token_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Process a list of tokens for frontend display.
    
    Args:
        tokens: List of token strings
        token_ids: Optional list of token IDs
        
    Returns:
        List of processed token data
    """
    if token_ids is None:
        token_ids = [None] * len(tokens)
    
    return [process_token(token, token_id) for token, token_id in zip(tokens, token_ids)]


def create_node_data(node_id: str, node_type: str, **kwargs) -> Dict[str, Any]:
    """Create standardized node data for the frontend.
    
    Args:
        node_id: Unique identifier for the node
        node_type: Type of the node (e.g., 'feature', 'token', 'logit')
        **kwargs: Additional node properties
        
    Returns:
        Standardized node data
    """
    node_data = {
        'id': node_id,
        'type': node_type,
        'label': kwargs.get('label', node_id),
        'value': kwargs.get('value', 0.0),
        'layer': kwargs.get('layer', 0),
        'position': kwargs.get('position', 0)
    }
    
    # Add any additional properties
    for key, value in kwargs.items():
        if key not in node_data:
            node_data[key] = value
    
    return node_data


def create_edge_data(source: str, target: str, weight: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create standardized edge data for the frontend.
    
    Args:
        source: Source node ID
        target: Target node ID
        weight: Edge weight/strength
        **kwargs: Additional edge properties
        
    Returns:
        Standardized edge data
    """
    edge_data = {
        'source': source,
        'target': target,
        'weight': weight,
        'type': kwargs.get('type', 'attribution')
    }
    
    # Add any additional properties
    for key, value in kwargs.items():
        if key not in edge_data:
            edge_data[key] = value
    
    return edge_data


def save_graph_json(graph_data: Dict[str, Any], output_path: Path) -> None:
    """Save graph data as JSON file.
    
    Args:
        graph_data: The graph data to save
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)


def load_graph_json(input_path: Path) -> Dict[str, Any]:
    """Load graph data from JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Loaded graph data
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)