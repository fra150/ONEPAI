#!/usr/bin/env python3
"""Script per la decodifica e analisi dei pattern di silenzio.

Questo script analizza i pattern di silenzio nei modelli AI,
identificando neuroni silenziosi, pensieri soppressi e decisioni non prese.
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Aggiungi il percorso src al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from onepai.core.silence_tracer import SilenceTracer
from onepai.analysis.silence_metrics import SilenceMetrics
from onepai.memory.unsaid_vault import UnsaidVault


def create_parser() -> argparse.ArgumentParser:
    """Crea il parser per gli argomenti dello script."""
    parser = argparse.ArgumentParser(
        description='Decodifica e analizza i pattern di silenzio nei modelli AI'
    )
    
    parser.add_argument(
        'model_path',
        help='Percorso al modello da analizzare'
    )
    
    parser.add_argument(
        '--input-data',
        help='File con i dati di input per l\'analisi'
    )
    
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=0.01,
        help='Soglia per identificare neuroni silenziosi (default: 0.01)'
    )
    
    parser.add_argument(
        '--analysis-depth',
        choices=['surface', 'deep', 'comprehensive'],
        default='deep',
        help='Profondità dell\'analisi (default: deep)'
    )
    
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        help='Layer specifici da analizzare (default: tutti)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='File di output per i risultati (default: silence_analysis.json)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'txt', 'csv'],
        default='json',
        help='Formato di output (default: json)'
    )
    
    parser.add_argument(
        '--include-patterns',
        action='store_true',
        help='Includi analisi dei pattern temporali'
    )
    
    parser.add_argument(
        '--include-correlations',
        action='store_true',
        help='Includi analisi delle correlazioni'
    )
    
    parser.add_argument(
        '--save-vault',
        help='Salva i contenuti non espressi in un vault'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Output verboso'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Genera visualizzazioni dei pattern di silenzio'
    )
    
    return parser


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Carica i dati di input per l'analisi."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                return json.load(f)
            elif input_file.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                # Assume text file with one input per line
                return [{'text': line.strip()} for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Errore nel caricamento dei dati di input: {e}")


def analyze_silence_patterns(model_path: str, input_data: List[Dict[str, Any]], 
                           silence_threshold: float, layers: List[int] = None,
                           analysis_depth: str = 'deep') -> Dict[str, Any]:
    """Analizza i pattern di silenzio nel modello."""
    tracer = SilenceTracer(model_path)
    metrics = SilenceMetrics()
    
    results = {
        'analysis_metadata': {
            'model_path': model_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'silence_threshold': silence_threshold,
            'analysis_depth': analysis_depth,
            'input_samples': len(input_data),
            'layers_analyzed': layers or 'all'
        },
        'silence_statistics': {},
        'silent_neurons': {},
        'suppressed_thoughts': [],
        'unexpressed_decisions': [],
        'silence_patterns': {}
    }
    
    # Analizza ogni input
    all_activations = []
    all_silence_maps = []
    
    for i, input_item in enumerate(input_data):
        if isinstance(input_item, dict) and 'text' in input_item:
            input_text = input_item['text']
        else:
            input_text = str(input_item)
        
        # Traccia il silenzio per questo input
        silence_map = tracer.trace_silence(input_text, layers=layers)
        activations = tracer.get_activations()
        
        all_activations.append(activations)
        all_silence_maps.append(silence_map)
        
        # Identifica pensieri soppressi
        suppressed = tracer.identify_suppressed_thoughts(silence_threshold)
        if suppressed:
            results['suppressed_thoughts'].extend([
                {
                    'input_index': i,
                    'input_text': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                    'suppressed_content': thought,
                    'layer': thought.get('layer'),
                    'confidence': thought.get('confidence', 0.0)
                }
                for thought in suppressed
            ])
        
        # Identifica decisioni non prese
        unexpressed = tracer.identify_unexpressed_decisions()
        if unexpressed:
            results['unexpressed_decisions'].extend([
                {
                    'input_index': i,
                    'input_text': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                    'decision_point': decision,
                    'alternative_paths': decision.get('alternatives', []),
                    'suppression_reason': decision.get('reason')
                }
                for decision in unexpressed
            ])
    
    # Calcola statistiche aggregate
    results['silence_statistics'] = calculate_silence_statistics(
        all_silence_maps, all_activations, silence_threshold
    )
    
    # Identifica neuroni costantemente silenziosi
    results['silent_neurons'] = identify_silent_neurons(
        all_activations, silence_threshold, layers
    )
    
    # Analizza pattern temporali se richiesto
    if analysis_depth in ['deep', 'comprehensive']:
        results['silence_patterns'] = analyze_temporal_patterns(
            all_silence_maps, all_activations
        )
    
    return results


def calculate_silence_statistics(silence_maps: List[Dict], activations: List[Dict], 
                               threshold: float) -> Dict[str, Any]:
    """Calcola statistiche sui pattern di silenzio."""
    stats = {
        'total_neurons_analyzed': 0,
        'silent_neuron_percentage': 0.0,
        'average_silence_duration': 0.0,
        'silence_distribution_by_layer': {},
        'silence_intensity_stats': {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    }
    
    if not silence_maps or not activations:
        return stats
    
    # Aggrega dati da tutti i campioni
    all_silence_values = []
    layer_silence_counts = {}
    
    for silence_map in silence_maps:
        for layer_name, layer_data in silence_map.items():
            if layer_name not in layer_silence_counts:
                layer_silence_counts[layer_name] = {'total': 0, 'silent': 0}
            
            if isinstance(layer_data, dict) and 'silence_mask' in layer_data:
                silence_mask = layer_data['silence_mask']
                if isinstance(silence_mask, (list, np.ndarray)):
                    silence_values = np.array(silence_mask)
                    all_silence_values.extend(silence_values.flatten())
                    
                    layer_silence_counts[layer_name]['total'] += len(silence_values.flatten())
                    layer_silence_counts[layer_name]['silent'] += np.sum(silence_values < threshold)
    
    if all_silence_values:
        all_silence_values = np.array(all_silence_values)
        stats['total_neurons_analyzed'] = len(all_silence_values)
        stats['silent_neuron_percentage'] = (np.sum(all_silence_values < threshold) / len(all_silence_values)) * 100
        
        stats['silence_intensity_stats'] = {
            'mean': float(np.mean(all_silence_values)),
            'std': float(np.std(all_silence_values)),
            'min': float(np.min(all_silence_values)),
            'max': float(np.max(all_silence_values))
        }
    
    # Calcola distribuzione per layer
    for layer_name, counts in layer_silence_counts.items():
        if counts['total'] > 0:
            stats['silence_distribution_by_layer'][layer_name] = {
                'total_neurons': counts['total'],
                'silent_neurons': counts['silent'],
                'silence_percentage': (counts['silent'] / counts['total']) * 100
            }
    
    return stats


def identify_silent_neurons(activations: List[Dict], threshold: float, 
                          layers: List[int] = None) -> Dict[str, Any]:
    """Identifica neuroni costantemente silenziosi."""
    silent_neurons = {
        'consistently_silent': {},
        'intermittently_silent': {},
        'silence_patterns': {}
    }
    
    if not activations:
        return silent_neurons
    
    # Aggrega attivazioni per neurone
    neuron_activations = {}
    
    for activation_set in activations:
        for layer_name, layer_activations in activation_set.items():
            if layers and not any(str(l) in layer_name for l in layers):
                continue
                
            if layer_name not in neuron_activations:
                neuron_activations[layer_name] = {}
            
            if isinstance(layer_activations, (list, np.ndarray)):
                layer_activations = np.array(layer_activations)
                for neuron_idx, activation in enumerate(layer_activations.flatten()):
                    if neuron_idx not in neuron_activations[layer_name]:
                        neuron_activations[layer_name][neuron_idx] = []
                    neuron_activations[layer_name][neuron_idx].append(float(activation))
    
    # Identifica neuroni silenziosi
    for layer_name, layer_neurons in neuron_activations.items():
        consistently_silent = []
        intermittently_silent = []
        
        for neuron_idx, activations_list in layer_neurons.items():
            activations_array = np.array(activations_list)
            
            # Neuroni costantemente silenziosi
            if np.all(activations_array < threshold):
                consistently_silent.append({
                    'neuron_index': neuron_idx,
                    'max_activation': float(np.max(activations_array)),
                    'mean_activation': float(np.mean(activations_array)),
                    'samples_analyzed': len(activations_array)
                })
            
            # Neuroni intermittentemente silenziosi
            elif np.mean(activations_array < threshold) > 0.7:  # 70% del tempo silenziosi
                intermittently_silent.append({
                    'neuron_index': neuron_idx,
                    'silence_ratio': float(np.mean(activations_array < threshold)),
                    'mean_activation': float(np.mean(activations_array)),
                    'activation_variance': float(np.var(activations_array))
                })
        
        if consistently_silent:
            silent_neurons['consistently_silent'][layer_name] = consistently_silent
        
        if intermittently_silent:
            silent_neurons['intermittently_silent'][layer_name] = intermittently_silent
    
    return silent_neurons


def analyze_temporal_patterns(silence_maps: List[Dict], activations: List[Dict]) -> Dict[str, Any]:
    """Analizza i pattern temporali del silenzio."""
    patterns = {
        'silence_sequences': [],
        'activation_cycles': {},
        'correlation_patterns': {},
        'emergence_patterns': {}
    }
    
    if len(silence_maps) < 2:
        return patterns
    
    # Analizza sequenze di silenzio
    for i in range(len(silence_maps) - 1):
        current_map = silence_maps[i]
        next_map = silence_maps[i + 1]
        
        # Identifica transizioni silenzio -> attivazione e viceversa
        transitions = identify_silence_transitions(current_map, next_map)
        if transitions:
            patterns['silence_sequences'].append({
                'sequence_index': i,
                'transitions': transitions
            })
    
    # Analizza cicli di attivazione
    patterns['activation_cycles'] = identify_activation_cycles(activations)
    
    return patterns


def identify_silence_transitions(map1: Dict, map2: Dict) -> List[Dict]:
    """Identifica transizioni nei pattern di silenzio."""
    transitions = []
    
    for layer_name in map1.keys():
        if layer_name in map2:
            # Confronta i pattern di silenzio tra i due stati
            if ('silence_mask' in map1[layer_name] and 
                'silence_mask' in map2[layer_name]):
                
                mask1 = np.array(map1[layer_name]['silence_mask'])
                mask2 = np.array(map2[layer_name]['silence_mask'])
                
                if mask1.shape == mask2.shape:
                    # Identifica neuroni che cambiano stato
                    silence_to_active = np.logical_and(mask1, ~mask2)
                    active_to_silence = np.logical_and(~mask1, mask2)
                    
                    if np.any(silence_to_active) or np.any(active_to_silence):
                        transitions.append({
                            'layer': layer_name,
                            'silence_to_active_count': int(np.sum(silence_to_active)),
                            'active_to_silence_count': int(np.sum(active_to_silence)),
                            'transition_ratio': float(np.sum(silence_to_active | active_to_silence) / len(mask1.flatten()))
                        })
    
    return transitions


def identify_activation_cycles(activations: List[Dict]) -> Dict[str, Any]:
    """Identifica cicli ricorrenti nelle attivazioni."""
    cycles = {
        'detected_cycles': [],
        'cycle_statistics': {}
    }
    
    # Implementazione semplificata per identificare pattern ciclici
    # In una implementazione completa, si potrebbero usare tecniche di analisi spettrale
    
    return cycles


def save_to_vault(results: Dict[str, Any], vault_path: str):
    """Salva i contenuti non espressi in un vault."""
    vault = UnsaidVault(vault_path)
    
    # Salva pensieri soppressi
    for thought in results.get('suppressed_thoughts', []):
        vault.store_unsaid_content(
            content_type='suppressed_thought',
            content=thought['suppressed_content'],
            metadata={
                'input_text': thought['input_text'],
                'layer': thought['layer'],
                'confidence': thought['confidence'],
                'timestamp': datetime.now().isoformat()
            }
        )
    
    # Salva decisioni non prese
    for decision in results.get('unexpressed_decisions', []):
        vault.store_unsaid_content(
            content_type='unexpressed_decision',
            content=decision['decision_point'],
            metadata={
                'input_text': decision['input_text'],
                'alternative_paths': decision['alternative_paths'],
                'suppression_reason': decision['suppression_reason'],
                'timestamp': datetime.now().isoformat()
            }
        )
    
    print(f"Contenuti non espressi salvati nel vault: {vault_path}")


def save_results(results: Dict[str, Any], output_file: str, format_type: str):
    """Salva i risultati nel formato specificato."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format_type == 'yaml':
        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    elif format_type == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            write_text_report(results, f)
    
    elif format_type == 'csv':
        write_csv_report(results, output_path)
    
    print(f"Risultati salvati: {output_path}")


def write_text_report(results: Dict[str, Any], file_handle):
    """Scrive un report testuale dei risultati."""
    metadata = results.get('analysis_metadata', {})
    stats = results.get('silence_statistics', {})
    
    file_handle.write("ANALISI DEI PATTERN DI SILENZIO\n")
    file_handle.write("=" * 50 + "\n\n")
    
    file_handle.write(f"Modello analizzato: {metadata.get('model_path', 'N/A')}\n")
    file_handle.write(f"Timestamp: {metadata.get('analysis_timestamp', 'N/A')}\n")
    file_handle.write(f"Campioni di input: {metadata.get('input_samples', 0)}\n")
    file_handle.write(f"Soglia di silenzio: {metadata.get('silence_threshold', 0.0)}\n\n")
    
    file_handle.write("STATISTICHE GENERALI\n")
    file_handle.write("-" * 20 + "\n")
    file_handle.write(f"Neuroni analizzati: {stats.get('total_neurons_analyzed', 0)}\n")
    file_handle.write(f"Percentuale neuroni silenziosi: {stats.get('silent_neuron_percentage', 0.0):.2f}%\n\n")
    
    # Aggiungi altre sezioni del report...
    suppressed_count = len(results.get('suppressed_thoughts', []))
    unexpressed_count = len(results.get('unexpressed_decisions', []))
    
    file_handle.write(f"Pensieri soppressi identificati: {suppressed_count}\n")
    file_handle.write(f"Decisioni non espresse identificate: {unexpressed_count}\n")


def write_csv_report(results: Dict[str, Any], output_path: Path):
    """Scrive un report CSV dei risultati."""
    import csv
    
    # Crea file CSV per neuroni silenziosi
    silent_neurons_file = output_path.with_suffix('.silent_neurons.csv')
    with open(silent_neurons_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Neuron_Index', 'Type', 'Max_Activation', 'Mean_Activation', 'Silence_Ratio'])
        
        silent_neurons = results.get('silent_neurons', {})
        
        for layer_name, neurons in silent_neurons.get('consistently_silent', {}).items():
            for neuron in neurons:
                writer.writerow([
                    layer_name,
                    neuron['neuron_index'],
                    'consistently_silent',
                    neuron['max_activation'],
                    neuron['mean_activation'],
                    1.0
                ])
        
        for layer_name, neurons in silent_neurons.get('intermittently_silent', {}).items():
            for neuron in neurons:
                writer.writerow([
                    layer_name,
                    neuron['neuron_index'],
                    'intermittently_silent',
                    'N/A',
                    neuron['mean_activation'],
                    neuron['silence_ratio']
                ])
    
    print(f"Report CSV neuroni silenziosi: {silent_neurons_file}")


def print_summary(results: Dict[str, Any]):
    """Stampa un riassunto dei risultati."""
    metadata = results.get('analysis_metadata', {})
    stats = results.get('silence_statistics', {})
    
    print("\n" + "=" * 60)
    print("RIASSUNTO ANALISI PATTERN DI SILENZIO")
    print("=" * 60)
    
    print(f"\nModello: {metadata.get('model_path', 'N/A')}")
    print(f"Campioni analizzati: {metadata.get('input_samples', 0)}")
    print(f"Neuroni totali: {stats.get('total_neurons_analyzed', 0)}")
    print(f"Neuroni silenziosi: {stats.get('silent_neuron_percentage', 0.0):.2f}%")
    
    suppressed_count = len(results.get('suppressed_thoughts', []))
    unexpressed_count = len(results.get('unexpressed_decisions', []))
    
    print(f"\nPensieri soppressi: {suppressed_count}")
    print(f"Decisioni non espresse: {unexpressed_count}")
    
    # Mostra distribuzione per layer
    layer_dist = stats.get('silence_distribution_by_layer', {})
    if layer_dist:
        print("\nDistribuzione silenzio per layer:")
        for layer_name, layer_stats in layer_dist.items():
            print(f"  {layer_name}: {layer_stats['silence_percentage']:.1f}% ({layer_stats['silent_neurons']}/{layer_stats['total_neurons']})")
    
    print("\n" + "=" * 60)


def main():
    """Funzione principale dello script."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Carica dati di input
        input_data = []
        if args.input_data:
            input_data = load_input_data(args.input_data)
            if args.verbose:
                print(f"Caricati {len(input_data)} campioni di input")
        else:
            # Usa dati di esempio se non specificati
            input_data = [
                {'text': 'What is the meaning of life?'},
                {'text': 'Explain quantum computing in simple terms.'},
                {'text': 'Should I invest in cryptocurrency?'}
            ]
            if args.verbose:
                print("Usando dati di esempio per l'analisi")
        
        # Esegui analisi
        if args.verbose:
            print(f"Avvio analisi del modello: {args.model_path}")
            print(f"Soglia di silenzio: {args.silence_threshold}")
            print(f"Profondità analisi: {args.analysis_depth}")
        
        results = analyze_silence_patterns(
            args.model_path,
            input_data,
            args.silence_threshold,
            args.layers,
            args.analysis_depth
        )
        
        # Salva nel vault se richiesto
        if args.save_vault:
            save_to_vault(results, args.save_vault)
        
        # Salva risultati
        output_file = args.output or f"silence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
        save_results(results, output_file, args.format)
        
        # Mostra riassunto
        if args.verbose:
            print_summary(results)
        else:
            stats = results.get('silence_statistics', {})
            print(f"Analisi completata. Neuroni silenziosi: {stats.get('silent_neuron_percentage', 0.0):.2f}%")
            print(f"Risultati salvati: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Errore durante l'analisi: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())