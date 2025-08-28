#!/usr/bin/env python3
"""
Script per eseguire analisi sul sistema ONEPAI
Utilizzo: python run_analysis.py --analysis <tipo> [--options]
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Aggiungi la directory src al path per importare il pacchetto onepai
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onepai.core.observer import ONEPAI_Observer
from onepai.core.archive import TreasureArchive
from onepai.reader.query import ShadowQueryEngine
from onepai.analysis.absence_patterns import AbsencePatternAnalyzer
from onepai.analysis.silence_metrics import SilenceMetricsCalculator
from onepai.analysis.void_statistics import VoidStatisticsCalculator
from onepai.utils.config import load_config
from onepai.utils.logger import setup_logger


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configura il parser degli argomenti della riga di comando"""
    parser = argparse.ArgumentParser(
        description="Strumento di analisi per il sistema ONEPAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argomenti principali
    parser.add_argument(
        "--analysis",
        type=str,
        required=True,
        choices=["shadow", "silence", "void", "absence", "treasure"],
        help="Tipo di analisi da eseguire"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Percorso del file di configurazione"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results",
        help="Directory di output per i risultati"
    )
    
    # Argomenti specifici per l'analisi shadow
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.1,
        help="Soglia per l'analisi delle ombre"
    )
    
    # Argomenti specifici per l'analisi silence
    parser.add_argument(
        "--silence-window",
        type=int,
        default=100,
        help="Dimensione della finestra per l'analisi del silenzio"
    )
    
    # Argomenti specifici per l'analisi void
    parser.add_argument(
        "--void-depth",
        type=int,
        default=3,
        help="Profondità dell'analisi del vuoto"
    )
    
    # Argomenti specifici per l'analisi absence
    parser.add_argument(
        "--absence-min-frequency",
        type=float,
        default=0.05,
        help="Frequenza minima per i pattern di assenza"
    )
    
    # Argomenti specifici per l'analisi treasure
    parser.add_argument(
        "--treasure-path",
        type=str,
        default="data/treasures",
        help="Percorso degli archivi treasure"
    )
    
    parser.add_argument(
        "--treasure-format",
        type=str,
        choices=["json", "csv", "parquet"],
        default="json",
        help="Formato di esportazione dei risultati"
    )
    
    return parser


def run_shadow_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue l'analisi delle ombre neurali"""
    logger.info("Avvio analisi shadow...")
    
    # Inizializza i componenti necessari
    observer = ONEPAI_Observer(target_ai=None)  # Sarà configurato tramite config
    query_engine = ShadowQueryEngine(config.get("shadow_query", {}))
    
    # Esegui l'analisi
    shadow_results = {
        "threshold": args.shadow_threshold,
        "patterns": observer.analyze_shadow_patterns(),
        "query_results": query_engine.query_shadows(
            threshold=args.shadow_threshold
        ),
        "statistics": query_engine.get_shadow_statistics()
    }
    
    logger.info("Analisi shadow completata")
    return shadow_results


def run_silence_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue l'analisi dei silenzi cognitivi"""
    logger.info("Avvio analisi silence...")
    
    # Inizializza i componenti necessari
    metrics_calc = SilenceMetricsCalculator(
        window_size=args.silence_window,
        config=config.get("silence_metrics", {})
    )
    
    # Esegui l'analisi
    silence_results = {
        "window_size": args.silence_window,
        "metrics": metrics_calc.calculate_metrics(),
        "patterns": metrics_calc.identify_silence_patterns(),
        "distribution": metrics_calc.get_silence_distribution()
    }
    
    logger.info("Analisi silence completata")
    return silence_results


def run_void_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue l'analisi del vuoto cognitivo"""
    logger.info("Avvio analisi void...")
    
    # Inizializza i componenti necessari
    void_calc = VoidStatisticsCalculator(
        depth=args.void_depth,
        config=config.get("void_statistics", {})
    )
    
    # Esegui l'analisi
    void_results = {
        "depth": args.void_depth,
        "statistics": void_calc.calculate_void_statistics(),
        "dimensions": void_calc.analyze_void_dimensions(),
        "correlations": void_calc.calculate_void_correlations()
    }
    
    logger.info("Analisi void completata")
    return void_results


def run_absence_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue l'analisi dei pattern di assenza"""
    logger.info("Avvio analisi absence...")
    
    # Inizializza i componenti necessari
    pattern_analyzer = AbsencePatternAnalyzer(
        min_frequency=args.absence_min_frequency,
        config=config.get("absence_patterns", {})
    )
    
    # Esegui l'analisi
    absence_results = {
        "min_frequency": args.absence_min_frequency,
        "patterns": pattern_analyzer.identify_patterns(),
        "clusters": pattern_analyzer.cluster_patterns(),
        "significance": pattern_analyzer.calculate_significance()
    }
    
    logger.info("Analisi absence completata")
    return absence_results


def run_treasure_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue l'analisi degli archivi treasure"""
    logger.info("Avvio analisi treasure...")
    
    # Inizializza i componenti necessari
    archive = TreasureArchive(
        base_path=args.treasure_path,
        config=config.get("archive", {})
    )
    
    # Esegui l'analisi
    treasure_results = {
        "archive_path": args.treasure_path,
        "statistics": archive.get_archive_statistics(),
        "fragments": archive.query_fragments(),
        "timeline": archive.get_activation_timeline(),
        "export_format": args.treasure_format
    }
    
    logger.info("Analisi treasure completata")
    return treasure_results


def save_results(results: Dict[str, Any], args: argparse.Namespace):
    """Salva i risultati dell'analisi"""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_type = args.analysis
    output_file = output_dir / f"{analysis_type}_analysis.{args.treasure_format}"
    
    logger.info(f"Salvataggio risultati in: {output_file}")
    
    if args.treasure_format == "json":
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif args.treasure_format == "csv":
        import pandas as pd
        # Converti i risultati in un formato tabellare
        df = pd.json_normalize(results)
        df.to_csv(output_file, index=False)
    
    elif args.treasure_format == "parquet":
        import pandas as pd
        df = pd.json_normalize(results)
        df.to_parquet(output_file, index=False)
    
    logger.info("Risultati salvati con successo")


def main():
    """Funzione principale"""
    # Configura il parser degli argomenti
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Carica la configurazione
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Errore nel caricamento della configurazione: {e}")
        sys.exit(1)
    
    # Configura il logger
    logger = setup_logger(config.get("logging", {}))
    
    logger.info(f"Avvio analisi ONEPAI: {args.analysis}")
    logger.info(f"Configurazione caricata da: {args.config}")
    
    try:
        # Esegui l'analisi specificata
        if args.analysis == "shadow":
            results = run_shadow_analysis(args, config)
        elif args.analysis == "silence":
            results = run_silence_analysis(args, config)
        elif args.analysis == "void":
            results = run_void_analysis(args, config)
        elif args.analysis == "absence":
            results = run_absence_analysis(args, config)
        elif args.analysis == "treasure":
            results = run_treasure_analysis(args, config)
        else:
            logger.error(f"Tipo di analisi non supportato: {args.analysis}")
            sys.exit(1)
        
        # Salva i risultati
        save_results(results, args)
        
        logger.info("Analisi completata con successo")
    
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione dell'analisi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()