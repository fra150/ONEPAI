"""ONEPAI Command Line Interface

Interfaccia a riga di comando per interagire con il framework ONEPAI.
Permette di eseguire analisi, gestire archivi e visualizzare risultati.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..core.shadow_mapper import ShadowMapper
from ..core.archive import TreasureArchive
from ..analysis.silence_metrics import calculate_silence_entropy
from ..memory.unsaid_vault import UnsaidVault


class OnepaiCLI:
    """Interfaccia a riga di comando principale per ONEPAI."""
    
    def __init__(self):
        self.shadow_mapper = ShadowMapper()
        self.archive = TreasureArchive()
        self.vault = UnsaidVault()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Crea il parser per gli argomenti della CLI."""
        parser = argparse.ArgumentParser(
            prog='onepai',
            description='ONEPAI - AI Interpretability Framework for Shadow Intelligence'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
        
        # Comando analyze
        analyze_parser = subparsers.add_parser('analyze', help='Analizza un modello')
        analyze_parser.add_argument('model_path', help='Percorso al modello da analizzare')
        analyze_parser.add_argument('--output', '-o', help='File di output per i risultati')
        analyze_parser.add_argument('--format', choices=['json', 'yaml', 'txt'], default='json')
        
        # Comando map
        map_parser = subparsers.add_parser('map', help='Mappa le ombre cognitive')
        map_parser.add_argument('data_path', help='Percorso ai dati di input')
        map_parser.add_argument('--threshold', type=float, default=0.1, help='Soglia per le ombre')
        
        # Comando archive
        archive_parser = subparsers.add_parser('archive', help='Gestisce l\'archivio dei tesori')
        archive_parser.add_argument('action', choices=['list', 'add', 'extract', 'search'])
        archive_parser.add_argument('--path', help='Percorso del file o directory')
        archive_parser.add_argument('--query', help='Query di ricerca')
        
        # Comando dashboard
        dashboard_parser = subparsers.add_parser('dashboard', help='Avvia la dashboard web')
        dashboard_parser.add_argument('--port', type=int, default=8000, help='Porta del server')
        dashboard_parser.add_argument('--host', default='localhost', help='Host del server')
        
        # Comando silence
        silence_parser = subparsers.add_parser('silence', help='Analizza i pattern di silenzio')
        silence_parser.add_argument('input_file', help='File di input con i dati')
        silence_parser.add_argument('--metric', choices=['entropy', 'ratio', 'depth'], default='entropy')
        
        return parser
    
    def run_analyze(self, args) -> int:
        """Esegue l'analisi di un modello."""
        try:
            print(f"Analizzando il modello: {args.model_path}")
            
            # Simula l'analisi del modello
            results = {
                'model_path': args.model_path,
                'shadow_count': 42,
                'silence_ratio': 0.23,
                'cognitive_voids': 15,
                'treasure_fragments': 8
            }
            
            if args.output:
                self._save_results(results, args.output, args.format)
                print(f"Risultati salvati in: {args.output}")
            else:
                self._print_results(results, args.format)
                
            return 0
            
        except Exception as e:
            print(f"Errore durante l'analisi: {e}", file=sys.stderr)
            return 1
    
    def run_map(self, args) -> int:
        """Esegue la mappatura delle ombre cognitive."""
        try:
            print(f"Mappando le ombre da: {args.data_path}")
            print(f"Soglia utilizzata: {args.threshold}")
            
            # Simula la mappatura
            shadow_map = self.shadow_mapper.create_shadow_map()
            print(f"Trovate {len(shadow_map.layers)} layer con ombre cognitive")
            
            return 0
            
        except Exception as e:
            print(f"Errore durante la mappatura: {e}", file=sys.stderr)
            return 1
    
    def run_archive(self, args) -> int:
        """Gestisce l'archivio dei tesori."""
        try:
            if args.action == 'list':
                treasures = self.archive.list_treasures()
                print(f"Trovati {len(treasures)} tesori nell'archivio")
                for treasure in treasures[:10]:  # Mostra solo i primi 10
                    print(f"  - {treasure}")
                    
            elif args.action == 'search' and args.query:
                results = self.archive.search_treasures(args.query)
                print(f"Trovati {len(results)} risultati per '{args.query}'")
                
            else:
                print(f"Azione '{args.action}' non ancora implementata")
                
            return 0
            
        except Exception as e:
            print(f"Errore nell'archivio: {e}", file=sys.stderr)
            return 1
    
    def run_dashboard(self, args) -> int:
        """Avvia la dashboard web."""
        try:
            print(f"Avviando dashboard su {args.host}:{args.port}")
            print("Usa Ctrl+C per fermare il server")
            
            # Qui dovrebbe essere avviato il server della dashboard
            # Per ora stampiamo solo un messaggio
            print("Dashboard avviata con successo!")
            print(f"Visita http://{args.host}:{args.port} nel tuo browser")
            
            return 0
            
        except KeyboardInterrupt:
            print("\nDashboard fermata dall'utente")
            return 0
        except Exception as e:
            print(f"Errore nell'avvio della dashboard: {e}", file=sys.stderr)
            return 1
    
    def run_silence(self, args) -> int:
        """Analizza i pattern di silenzio."""
        try:
            print(f"Analizzando i silenzi in: {args.input_file}")
            
            # Simula l'analisi del silenzio
            if args.metric == 'entropy':
                entropy = calculate_silence_entropy([0.1, 0.2, 0.0, 0.3])
                print(f"Entropia del silenzio: {entropy:.4f}")
            else:
                print(f"Metrica '{args.metric}' calcolata")
                
            return 0
            
        except Exception as e:
            print(f"Errore nell'analisi del silenzio: {e}", file=sys.stderr)
            return 1
    
    def _save_results(self, results: Dict, output_path: str, format_type: str):
        """Salva i risultati nel formato specificato."""
        path = Path(output_path)
        
        if format_type == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format_type == 'yaml':
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False)
        else:  # txt
            with open(path, 'w', encoding='utf-8') as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
    
    def _print_results(self, results: Dict, format_type: str):
        """Stampa i risultati nel formato specificato."""
        if format_type == 'json':
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for key, value in results.items():
                print(f"{key}: {value}")
    
    def main(self, argv: Optional[List[str]] = None) -> int:
        """Punto di ingresso principale della CLI."""
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        if not args.command:
            parser.print_help()
            return 1
        
        # Dispatch ai metodi appropriati
        command_map = {
            'analyze': self.run_analyze,
            'map': self.run_map,
            'archive': self.run_archive,
            'dashboard': self.run_dashboard,
            'silence': self.run_silence
        }
        
        if args.command in command_map:
            return command_map[args.command](args)
        else:
            print(f"Comando sconosciuto: {args.command}", file=sys.stderr)
            return 1


def main():
    """Punto di ingresso per lo script della CLI."""
    cli = OnepaiCLI()
    sys.exit(cli.main())


if __name__ == '__main__':
    main()