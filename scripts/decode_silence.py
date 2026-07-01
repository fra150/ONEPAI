#!/usr/bin/env python3
"""Script per la gestione degli archivi ONEPAI.

Questo script fornisce funzionalità per gestire archivi di tesori,
silence patterns, shadow maps e altri artefatti cognitivi.
"""

import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Aggiungi il percorso src al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from onepai.core.archive import TreasureArchive
from onepai.core.crypto import CryptoManager
from onepai.memory.unsaid_vault import UnsaidVault


def create_parser() -> argparse.ArgumentParser:
    """Crea il parser per gli argomenti dello script."""
    parser = argparse.ArgumentParser(
        description='Gestisce gli archivi ONEPAI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Comando list
    list_parser = subparsers.add_parser('list', help='Elenca contenuti degli archivi')
    list_parser.add_argument(
        '--archive-type',
        choices=['treasures', 'shadows', 'silences', 'voids', 'all'],
        default='all',
        help='Tipo di archivio da elencare'
    )
    list_parser.add_argument(
        '--format',
        choices=['table', 'json', 'summary'],
        default='table',
        help='Formato di output'
    )
    list_parser.add_argument(
        '--filter-by',
        help='Filtra per tag, tipo o data (formato: campo:valore)'
    )
    
    # Comando backup
    backup_parser = subparsers.add_parser('backup', help='Crea backup degli archivi')
    backup_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory di destinazione per il backup'
    )
    backup_parser.add_argument(
        '--compress',
        action='store_true',
        help='Comprimi il backup'
    )
    backup_parser.add_argument(
        '--encrypt',
        action='store_true',
        help='Cripta il backup'
    )
    backup_parser.add_argument(
        '--password',
        help='Password per la crittografia'
    )
    
    # Comando restore
    restore_parser = subparsers.add_parser('restore', help='Ripristina da backup')
    restore_parser.add_argument(
        'backup_path',
        help='Percorso del backup da ripristinare'
    )
    restore_parser.add_argument(
        '--password',
        help='Password per decrittare il backup'
    )
    restore_parser.add_argument(
        '--force',
        action='store_true',
        help='Sovrascrivi file esistenti'
    )
    
    # Comando clean
    clean_parser = subparsers.add_parser('clean', help='Pulisce archivi obsoleti')
    clean_parser.add_argument(
        '--older-than',
        type=int,
        default=30,
        help='Rimuovi file più vecchi di N giorni (default: 30)'
    )
    clean_parser.add_argument(
        '--archive-type',
        choices=['treasures', 'shadows', 'silences', 'voids'],
        help='Tipo di archivio da pulire (default: tutti)'
    )
    clean_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostra cosa verrebbe rimosso senza rimuoverlo'
    )
    
    # Comando export
    export_parser = subparsers.add_parser('export', help='Esporta archivi')
    export_parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'csv', 'xml'],
        default='json',
        help='Formato di esportazione'
    )
    export_parser.add_argument(
        '--output',
        required=True,
        help='File di output'
    )
    export_parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Includi metadati dettagliati'
    )
    
    # Comando import
    import_parser = subparsers.add_parser('import', help='Importa archivi')
    import_parser.add_argument(
        'import_file',
        help='File da importare'
    )
    import_parser.add_argument(
        '--merge',
        action='store_true',
        help='Unisci con archivi esistenti'
    )
    
    # Comando verify
    verify_parser = subparsers.add_parser('verify', help='Verifica integrità archivi')
    verify_parser.add_argument(
        '--fix',
        action='store_true',
        help='Tenta di riparare errori trovati'
    )
    
    # Comando stats
    stats_parser = subparsers.add_parser('stats', help='Mostra statistiche archivi')
    stats_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Mostra statistiche dettagliate'
    )
    
    # Argomenti globali
    parser.add_argument(
        '--data-dir',
        default=str(project_root / "data"),
        help='Directory dei dati (default: ./data)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Output verboso'
    )
    
    return parser


class ArchiveManager:
    """Gestore degli archivi ONEPAI."""
    
    def __init__(self, data_dir: str, verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.crypto = CryptoManager()
        
        # Inizializza le directory degli archivi
        self.archives = {
            'treasures': self.data_dir / 'treasures',
            'shadows': self.data_dir / 'shadows',
            'silences': self.data_dir / 'silences',
            'voids': self.data_dir / 'voids'
        }
        
        # Crea le directory se non esistono
        for archive_path in self.archives.values():
            archive_path.mkdir(parents=True, exist_ok=True)
    
    def list_archives(self, archive_type: str = 'all', format_type: str = 'table', 
                     filter_by: str = None) -> Dict[str, Any]:
        """Elenca i contenuti degli archivi."""
        results = {}
        
        archives_to_list = [archive_type] if archive_type != 'all' else list(self.archives.keys())
        
        for arch_type in archives_to_list:
            if arch_type in self.archives:
                archive_path = self.archives[arch_type]
                files = self._scan_archive(archive_path, filter_by)
                results[arch_type] = files
        
        return results
    
    def _scan_archive(self, archive_path: Path, filter_by: str = None) -> List[Dict[str, Any]]:
        """Scansiona un archivio e restituisce informazioni sui file."""
        files = []
        
        if not archive_path.exists():
            return files
        
        for file_path in archive_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_info = self._get_file_info(file_path)
                
                # Applica filtro se specificato
                if filter_by and not self._matches_filter(file_info, filter_by):
                    continue
                
                files.append(file_info)
        
        return sorted(files, key=lambda x: x['modified_time'], reverse=True)
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Ottiene informazioni dettagliate su un file."""
        stat = file_path.stat()
        
        info = {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': file_path.suffix,
            'type': self._determine_file_type(file_path)
        }
        
        # Tenta di estrarre metadati se è un file JSON
        if file_path.suffix == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'metadata' in data:
                        metadata = data['metadata']
                        info.update({
                            'treasure_type': metadata.get('type'),
                            'significance': metadata.get('significance'),
                            'tags': metadata.get('tags', []),
                            'source': metadata.get('source')
                        })
            except Exception:
                pass  # Ignora errori di parsing
        
        return info
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determina il tipo di file basandosi su estensione e contenuto."""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.json': 'treasure',
            '.encrypted': 'encrypted_treasure',
            '.shadow': 'shadow_map',
            '.silence': 'silence_pattern',
            '.void': 'void_analysis',
            '.key': 'encryption_key',
            '.backup': 'backup_file'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    def _matches_filter(self, file_info: Dict[str, Any], filter_by: str) -> bool:
        """Verifica se un file corrisponde al filtro specificato."""
        if ':' not in filter_by:
            return False
        
        field, value = filter_by.split(':', 1)
        
        if field == 'type':
            return file_info.get('treasure_type') == value or file_info.get('type') == value
        elif field == 'tag':
            return value in file_info.get('tags', [])
        elif field == 'date':
            # Formato: YYYY-MM-DD
            file_date = file_info['modified_time'][:10]
            return file_date == value
        elif field == 'size':
            # Formato: >1000, <500, =1024
            operator = value[0]
            size_value = int(value[1:])
            file_size = file_info['size']
            
            if operator == '>':
                return file_size > size_value
            elif operator == '<':
                return file_size < size_value
            elif operator == '=':
                return file_size == size_value
        
        return False
    
    def create_backup(self, output_dir: str, compress: bool = False, 
                     encrypt: bool = False, password: str = None) -> str:
        """Crea un backup degli archivi."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"onepai_backup_{timestamp}"
        
        if compress:
            backup_file = output_path / f"{backup_name}.tar.gz"
            self._create_compressed_backup(backup_file)
        else:
            backup_dir = output_path / backup_name
            self._create_directory_backup(backup_dir)
            backup_file = backup_dir
        
        if encrypt:
            encrypted_file = self._encrypt_backup(backup_file, password)
            if backup_file != encrypted_file:
                # Rimuovi il file non crittografato se è stato creato un file crittografato
                if backup_file.is_file():
                    backup_file.unlink()
                elif backup_file.is_dir():
                    shutil.rmtree(backup_file)
            backup_file = encrypted_file
        
        return str(backup_file)
    
    def _create_compressed_backup(self, backup_file: Path):
        """Crea un backup compresso."""
        import tarfile
        
        with tarfile.open(backup_file, 'w:gz') as tar:
            for archive_name, archive_path in self.archives.items():
                if archive_path.exists():
                    tar.add(archive_path, arcname=archive_name)
        
        if self.verbose:
            print(f"Backup compresso creato: {backup_file}")
    
    def _create_directory_backup(self, backup_dir: Path):
        """Crea un backup in directory."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for archive_name, archive_path in self.archives.items():
            if archive_path.exists():
                dest_path = backup_dir / archive_name
                shutil.copytree(archive_path, dest_path, dirs_exist_ok=True)
        
        # Crea file di metadati del backup
        metadata = {
            'backup_timestamp': datetime.now().isoformat(),
            'onepai_version': '1.0.0',
            'archives_included': list(self.archives.keys()),
            'total_files': sum(len(list(p.rglob('*'))) for p in self.archives.values() if p.exists())
        }
        
        with open(backup_dir / 'backup_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Backup directory creato: {backup_dir}")
    
    def _encrypt_backup(self, backup_file: Path, password: str = None) -> Path:
        """Cripta un backup."""
        if password:
            self.crypto.derive_key_from_password(password)
        else:
            self.crypto.generate_key()
            password = f"auto_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Leggi il file di backup
        if backup_file.is_file():
            with open(backup_file, 'rb') as f:
                data = f.read()
        else:
            # Se è una directory, comprimi prima
            import tarfile
            import io
            
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
                tar.add(backup_file, arcname=backup_file.name)
            data = buffer.getvalue()
        
        # Cripta i dati
        encrypted_data = self.crypto.encrypt_data(data)
        
        # Salva il file crittografato
        encrypted_file = backup_file.with_suffix(backup_file.suffix + '.encrypted')
        with open(encrypted_file, 'wb') as f:
            f.write(encrypted_data)
        
        # Salva la password
        password_file = backup_file.with_suffix('.key')
        with open(password_file, 'w') as f:
            f.write(password)
        
        if self.verbose:
            print(f"Backup crittografato: {encrypted_file}")
            print(f"Chiave salvata: {password_file}")
        
        return encrypted_file
    
    def restore_backup(self, backup_path: str, password: str = None, force: bool = False) -> bool:
        """Ripristina da un backup."""
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"File di backup non trovato: {backup_path}")
        
        # Verifica se esistono già archivi e se force non è specificato
        if not force:
            existing_files = []
            for archive_path in self.archives.values():
                if archive_path.exists() and any(archive_path.iterdir()):
                    existing_files.append(str(archive_path))
            
            if existing_files:
                print("ATTENZIONE: I seguenti archivi contengono già dati:")
                for path in existing_files:
                    print(f"  - {path}")
                print("Usa --force per sovrascrivere o sposta i dati esistenti.")
                return False
        
        # Decrittografa se necessario
        if backup_file.suffix == '.encrypted':
            backup_file = self._decrypt_backup(backup_file, password)
        
        # Ripristina il backup
        if backup_file.suffix in ['.tar', '.gz'] or backup_file.name.endswith('.tar.gz'):
            self._restore_compressed_backup(backup_file)
        else:
            self._restore_directory_backup(backup_file)
        
        if self.verbose:
            print(f"Backup ripristinato da: {backup_path}")
        
        return True
    
    def _decrypt_backup(self, encrypted_file: Path, password: str = None) -> Path:
        """Decrittografa un backup."""
        if not password:
            # Cerca il file della password
            key_file = encrypted_file.with_suffix('.key')
            if key_file.exists():
                with open(key_file, 'r') as f:
                    password = f.read().strip()
            else:
                raise ValueError("Password richiesta per decrittografare il backup")
        
        self.crypto.derive_key_from_password(password)
        
        # Decrittografa
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.crypto.decrypt_data(encrypted_data)
        
        # Salva il file decrittografato temporaneamente
        temp_file = encrypted_file.with_suffix('.temp')
        with open(temp_file, 'wb') as f:
            f.write(decrypted_data)
        
        return temp_file
    
    def _restore_compressed_backup(self, backup_file: Path):
        """Ripristina da un backup compresso."""
        import tarfile
        
        with tarfile.open(backup_file, 'r:*') as tar:
            tar.extractall(self.data_dir)
    
    def _restore_directory_backup(self, backup_dir: Path):
        """Ripristina da un backup directory."""
        for archive_name in self.archives.keys():
            source_path = backup_dir / archive_name
            if source_path.exists():
                dest_path = self.archives[archive_name]
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
    
    def clean_archives(self, older_than_days: int = 30, archive_type: str = None, 
                      dry_run: bool = False) -> Dict[str, List[str]]:
        """Pulisce archivi obsoleti."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_files = {}
        
        archives_to_clean = [archive_type] if archive_type else list(self.archives.keys())
        
        for arch_type in archives_to_clean:
            if arch_type in self.archives:
                archive_path = self.archives[arch_type]
                files_to_remove = []
                
                for file_path in archive_path.rglob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            files_to_remove.append(str(file_path))
                            if not dry_run:
                                file_path.unlink()
                
                removed_files[arch_type] = files_to_remove
        
        return removed_files
    
    def export_archives(self, format_type: str = 'json', output_file: str = None, 
                       include_metadata: bool = False) -> str:
        """Esporta gli archivi nel formato specificato."""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"onepai_export_{timestamp}.{format_type}"
        
        # Raccoglie tutti i dati
        export_data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'onepai_version': '1.0.0',
                'format': format_type,
                'include_metadata': include_metadata
            },
            'archives': {}
        }
        
        for arch_type, archive_path in self.archives.items():
            archive_data = []
            
            for file_path in archive_path.rglob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if include_metadata:
                            file_info = self._get_file_info(file_path)
                            data['_file_metadata'] = file_info
                        
                        archive_data.append(data)
                except Exception as e:
                    if self.verbose:
                        print(f"Errore nel leggere {file_path}: {e}")
            
            export_data['archives'][arch_type] = archive_data
        
        # Salva nel formato richiesto
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'yaml':
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
        
        elif format_type == 'csv':
            self._export_to_csv(export_data, output_path)
        
        elif format_type == 'xml':
            self._export_to_xml(export_data, output_path)
        
        return str(output_path)
    
    def _export_to_csv(self, data: Dict[str, Any], output_path: Path):
        """Esporta in formato CSV."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Archive', 'Type', 'Name', 'Created', 'Significance', 'Tags'])
            
            for arch_type, items in data['archives'].items():
                for item in items:
                    metadata = item.get('metadata', {})
                    writer.writerow([
                        arch_type,
                        metadata.get('type', ''),
                        metadata.get('name', ''),
                        metadata.get('created_at', ''),
                        metadata.get('significance', ''),
                        ', '.join(metadata.get('tags', []))
                    ])
    
    def _export_to_xml(self, data: Dict[str, Any], output_path: Path):
        """Esporta in formato XML."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('onepai_export')
        
        # Metadati
        metadata_elem = ET.SubElement(root, 'metadata')
        for key, value in data['export_metadata'].items():
            elem = ET.SubElement(metadata_elem, key)
            elem.text = str(value)
        
        # Archivi
        archives_elem = ET.SubElement(root, 'archives')
        for arch_type, items in data['archives'].items():
            archive_elem = ET.SubElement(archives_elem, 'archive', type=arch_type)
            
            for item in items:
                item_elem = ET.SubElement(archive_elem, 'item')
                self._dict_to_xml(item, item_elem)
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element):
        """Converte un dizionario in XML."""
        for key, value in data.items():
            elem = ET.SubElement(parent, key)
            
            if isinstance(value, dict):
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                for item in value:
                    item_elem = ET.SubElement(elem, 'item')
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                elem.text = str(value)
    
    def verify_archives(self, fix: bool = False) -> Dict[str, Any]:
        """Verifica l'integrità degli archivi."""
        results = {
            'total_files': 0,
            'corrupted_files': [],
            'missing_metadata': [],
            'orphaned_keys': [],
            'fixed_issues': []
        }
        
        for arch_type, archive_path in self.archives.items():
            if not archive_path.exists():
                continue
            
            for file_path in archive_path.rglob('*'):
                if file_path.is_file():
                    results['total_files'] += 1
                    
                    # Verifica file JSON
                    if file_path.suffix == '.json':
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                # Verifica presenza metadati
                                if 'metadata' not in data:
                                    results['missing_metadata'].append(str(file_path))
                                    
                                    if fix:
                                        # Aggiungi metadati di base
                                        data['metadata'] = {
                                            'id': file_path.stem,
                                            'created_at': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                                            'type': 'unknown',
                                            'version': '1.0'
                                        }
                                        
                                        with open(file_path, 'w', encoding='utf-8') as f:
                                            json.dump(data, f, indent=2, ensure_ascii=False)
                                        
                                        results['fixed_issues'].append(f"Added metadata to {file_path}")
                        
                        except json.JSONDecodeError:
                            results['corrupted_files'].append(str(file_path))
                    
                    # Verifica file di chiavi orfani
                    elif file_path.suffix == '.key':
                        corresponding_file = file_path.with_suffix('.encrypted')
                        if not corresponding_file.exists():
                            results['orphaned_keys'].append(str(file_path))
                            
                            if fix:
                                file_path.unlink()
                                results['fixed_issues'].append(f"Removed orphaned key {file_path}")
        
        return results
    
    def get_statistics(self, detailed: bool = False) -> Dict[str, Any]:
        """Ottiene statistiche sugli archivi."""
        stats = {
            'summary': {
                'total_archives': len(self.archives),
                'total_files': 0,
                'total_size_bytes': 0,
                'oldest_file': None,
                'newest_file': None
            },
            'by_archive': {}
        }
        
        oldest_time = None
        newest_time = None
        
        for arch_type, archive_path in self.archives.items():
            archive_stats = {
                'file_count': 0,
                'total_size': 0,
                'file_types': {},
                'avg_file_size': 0
            }
            
            if detailed:
                archive_stats['files'] = []
            
            if archive_path.exists():
                file_sizes = []
                
                for file_path in archive_path.rglob('*'):
                    if file_path.is_file():
                        file_stat = file_path.stat()
                        file_size = file_stat.st_size
                        file_time = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        archive_stats['file_count'] += 1
                        archive_stats['total_size'] += file_size
                        file_sizes.append(file_size)
                        
                        # Traccia tipo di file
                        file_type = self._determine_file_type(file_path)
                        archive_stats['file_types'][file_type] = archive_stats['file_types'].get(file_type, 0) + 1
                        
                        # Traccia file più vecchio e più nuovo
                        if oldest_time is None or file_time < oldest_time:
                            oldest_time = file_time
                            stats['summary']['oldest_file'] = str(file_path)
                        
                        if newest_time is None or file_time > newest_time:
                            newest_time = file_time
                            stats['summary']['newest_file'] = str(file_path)
                        
                        if detailed:
                            archive_stats['files'].append({
                                'name': file_path.name,
                                'size': file_size,
                                'type': file_type,
                                'modified': file_time.isoformat()
                            })
                
                if file_sizes:
                    archive_stats['avg_file_size'] = sum(file_sizes) / len(file_sizes)
            
            stats['by_archive'][arch_type] = archive_stats
            stats['summary']['total_files'] += archive_stats['file_count']
            stats['summary']['total_size_bytes'] += archive_stats['total_size']
        
        return stats


def format_output(data: Any, format_type: str):
    """Formatta l'output nel formato richiesto."""
    if format_type == 'json':
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    
    elif format_type == 'table':
        if isinstance(data, dict):
            for archive_type, files in data.items():
                print(f"\n=== {archive_type.upper()} ===")
                if files:
                    print(f"{'Nome':<30} {'Tipo':<15} {'Dimensione':<10} {'Modificato':<20}")
                    print("-" * 75)
                    
                    for file_info in files:
                        size_str = f"{file_info['size']:,} B"
                        modified_str = file_info['modified_time'][:19].replace('T', ' ')
                        print(f"{file_info['name']:<30} {file_info.get('type', 'unknown'):<15} {size_str:<10} {modified_str:<20}")
                else:
                    print("Nessun file trovato.")
    
    elif format_type == 'summary':
        if isinstance(data, dict):
            total_files = sum(len(files) for files in data.values())
            print(f"Totale archivi: {len(data)}")
            print(f"Totale file: {total_files}")
            
            for archive_type, files in data.items():
                print(f"  {archive_type}: {len(files)} file")


def main():
    """Funzione principale dello script."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        manager = ArchiveManager(args.data_dir, args.verbose)
        
        if args.command == 'list':
            results = manager.list_archives(args.archive_type, args.format, args.filter_by)
            format_output(results, args.format)
        
        elif args.command == 'backup':
            backup_file = manager.create_backup(
                args.output_dir, args.compress, args.encrypt, args.password
            )
            print(f"Backup creato: {backup_file}")
        
        elif args.command == 'restore':
            success = manager.restore_backup(args.backup_path, args.password, args.force)
            if success:
                print("Backup ripristinato con successo")
            else:
                print("Ripristino annullato")
                return 1
        
        elif args.command == 'clean':
            removed = manager.clean_archives(args.older_than, args.archive_type, args.dry_run)
            
            total_removed = sum(len(files) for files in removed.values())
            action = "Sarebbero rimossi" if args.dry_run else "Rimossi"
            print(f"{action} {total_removed} file:")
            
            for archive_type, files in removed.items():
                if files:
                    print(f"  {archive_type}: {len(files)} file")
                    if args.verbose:
                        for file_path in files:
                            print(f"    - {file_path}")
        
        elif args.command == 'export':
            export_file = manager.export_archives(args.format, args.output, args.include_metadata)
            print(f"Archivi esportati: {export_file}")
        
        elif args.command == 'verify':
            results = manager.verify_archives(args.fix)
            
            print(f"File totali verificati: {results['total_files']}")
            print(f"File corrotti: {len(results['corrupted_files'])}")
            print(f"File senza metadati: {len(results['missing_metadata'])}")
            print(f"Chiavi orfane: {len(results['orphaned_keys'])}")
            
            if args.fix:
                print(f"Problemi risolti: {len(results['fixed_issues'])}")
                if args.verbose and results['fixed_issues']:
                    for fix in results['fixed_issues']:
                        print(f"  - {fix}")
        
        elif args.command == 'stats':
            stats = manager.get_statistics(args.detailed)
            
            summary = stats['summary']
            print(f"Statistiche Archivi ONEPAI")
            print(f"=" * 30)
            print(f"Archivi totali: {summary['total_archives']}")
            print(f"File totali: {summary['total_files']}")
            print(f"Dimensione totale: {summary['total_size_bytes']:,} bytes")
            
            if summary['oldest_file']:
                print(f"File più vecchio: {summary['oldest_file']}")
            if summary['newest_file']:
                print(f"File più recente: {summary['newest_file']}")
            
            print("\nPer archivio:")
            for arch_type, arch_stats in stats['by_archive'].items():
                print(f"  {arch_type}:")
                print(f"    File: {arch_stats['file_count']}")
                print(f"    Dimensione: {arch_stats['total_size']:,} bytes")
                if arch_stats['file_count'] > 0:
                    print(f"    Dimensione media: {arch_stats['avg_file_size']:.0f} bytes")
                
                if args.detailed and arch_stats['file_types']:
                    print(f"    Tipi di file:")
                    for file_type, count in arch_stats['file_types'].items():
                        print(f"      {file_type}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())