"""
Archivio del non-detto - Componente per memorizzare e gestire i contenuti
che sono stati generati ma non comunicati dall'IA.
"""

import json
import pickle
import hashlib
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import uuid

@dataclass
class UnsaidItem:
    """Rappresenta un contenuto generato ma non comunicato"""
    id: str
    timestamp: datetime
    content_type: str  # 'text', 'image', 'audio', etc.
    content: Any
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    confidence: float  # Livello di confidenza che questo contenuto fosse pertinente
    rejection_reason: Optional[str] = None

class UnsaidVault:
    """
    Sistema per archiviare e gestire i contenuti che sono stati generati
    internamente dall'IA ma non sono stati comunicati all'esterno.
    """
    
    def __init__(self, storage_path: str = "data/treasures/unsaid_vault"):
        """
        Inizializza l'archivio del non-detto.
        
        Args:
            storage_path: Percorso dove salvare i dati dell'archivio
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.storage_path / "index.pkl"
        self.items_path = self.storage_path / "items"
        self.items_path.mkdir(exist_ok=True)
        
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """
        Carica l'indice degli elementi non-detti dal disco.
        
        Returns:
            Dizionario indicizzato per ID
        """
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Errore nel caricamento dell'indice: {e}")
        
        return {}
    
    def _save_index(self):
        """Salva l'indice degli elementi non-detti su disco."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"Errore nel salvataggio dell'indice: {e}")
    
    def add_unsaid_item(self, content_type: str, content: Any, 
                       metadata: Dict[str, Any], context: Dict[str, Any],
                       confidence: float, rejection_reason: Optional[str] = None) -> str:
        """
        Aggiunge un nuovo elemento all'archivio del non-detto.
        
        Args:
            content_type: Tipo di contenuto ('text', 'image', etc.)
            content: Il contenuto effettivo
            metadata: Metadati del contenuto
            context: Contesto in cui è stato generato
            confidence: Livello di confidenza (0-1)
            rejection_reason: Motivo della non-comunicazione
            
        Returns:
            ID univoco dell'elemento aggiunto
        """
        # Genera un ID univoco
        item_id = str(uuid.uuid4())
        
        # Crea l'oggetto elemento
        item = UnsaidItem(
            id=item_id,
            timestamp=datetime.now(),
            content_type=content_type,
            content=content,
            metadata=metadata,
            context=context,
            confidence=confidence,
            rejection_reason=rejection_reason
        )
        
        # Salva il contenuto su disco
        self._save_item_content(item)
        
        # Aggiorna l'indice
        self.index[item_id] = {
            "timestamp": item.timestamp,
            "content_type": item.content_type,
            "content_hash": self._compute_content_hash(content),
            "metadata": metadata,
            "context": context,
            "confidence": confidence,
            "rejection_reason": rejection_reason,
            "file_path": self._get_item_path(item_id)
        }
        
        # Salva l'indice aggiornato
        self._save_index()
        
        return item_id
    
    def _save_item_content(self, item: UnsaidItem):
        """
        Salva il contenuto di un elemento su disco.
        
        Args:
            item: Elemento da salvare
        """
        file_path = self._get_item_path(item.id)
        
        try:
            if item.content_type == 'text':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(item.content)
            elif item.content_type in ['image', 'audio', 'video']:
                # Per contenuti binari, usa pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(item.content, f)
            else:
                # Per altri tipi, usa JSON se possibile, altrimenti pickle
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(item.content, f, ensure_ascii=False, indent=2)
                except (TypeError, OverflowError):
                    with open(file_path, 'wb') as f:
                        pickle.dump(item.content, f)
        except Exception as e:
            print(f"Errore nel salvataggio del contenuto {item.id}: {e}")
    
    def _get_item_path(self, item_id: str) -> Path:
        """
        Ottiene il percorso del file per un elemento dato il suo ID.
        
        Args:
            item_id: ID dell'elemento
            
        Returns:
            Percorso del file
        """
        return self.items_path / f"{item_id}.dat"
    
    def _compute_content_hash(self, content: Any) -> str:
        """
        Calcola un hash del contenuto per identificazione univoca.
        
        Args:
            content: Contenuto di cui calcolare l'hash
            
        Returns:
            Stringa hash
        """
        try:
            # Prova a serializzare in JSON
            content_str = json.dumps(content, sort_keys=True)
        except (TypeError, OverflowError):
            # Se non è possibile, usa pickle
            content_bytes = pickle.dumps(content)
            content_str = content_bytes.decode('latin1')
        
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def get_unsaid_item(self, item_id: str) -> Optional[UnsaidItem]:
        """
        Recupera un elemento dall'archivio dato il suo ID.
        
        Args:
            item_id: ID dell'elemento
            
        Returns:
            L'elemento richiesto o None se non trovato
        """
        if item_id not in self.index:
            return None
        
        index_entry = self.index[item_id]
        file_path = Path(index_entry["file_path"])
        
        if not file_path.exists():
            return None
        
        try:
            # Carica il contenuto dal file
            content = self._load_item_content(file_path, index_entry["content_type"])
            
            # Ricostruisci l'oggetto elemento
            item = UnsaidItem(
                id=item_id,
                timestamp=index_entry["timestamp"],
                content_type=index_entry["content_type"],
                content=content,
                metadata=index_entry["metadata"],
                context=index_entry["context"],
                confidence=index_entry["confidence"],
                rejection_reason=index_entry["rejection_reason"]
            )
            
            return item
        except Exception as e:
            print(f"Errore nel caricamento dell'elemento {item_id}: {e}")
            return None
    
    def _load_item_content(self, file_path: Path, content_type: str) -> Any:
        """
        Carica il contenuto di un elemento dal disco.
        
        Args:
            file_path: Percorso del file
            content_type: Tipo di contenuto
            
        Returns:
            Il contenuto caricato
        """
        if content_type == 'text':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif content_type in ['image', 'audio', 'video']:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Prova prima JSON, poi pickle
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
    
    def query_unsaid_items(self, filters: Dict[str, Any] = None, 
                          limit: Optional[int] = None) -> List[UnsaidItem]:
        """
        Interroga l'archivio per elementi che corrispondono a filtri specifici.
        
        Args:
            filters: Dizionario di filtri da applicare
            limit: Numero massimo di risultati
            
        Returns:
            Lista di elementi che corrispondono ai filtri
        """
        results = []
        
        for item_id, index_entry in self.index.items():
            # Applica i filtri
            if filters and not self._item_matches_filters(index_entry, filters):
                continue
            
            # Carica l'elemento completo
            item = self.get_unsaid_item(item_id)
            if item:
                results.append(item)
                
                # Controlla il limite
                if limit is not None and len(results) >= limit:
                    break
        
        # Ordina per timestamp (più recenti prima)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def _item_matches_filters(self, index_entry: Dict[str, Any], 
                             filters: Dict[str, Any]) -> bool:
        """
        Verifica se una voce di indice corrisponde ai filtri specificati.
        
        Args:
            index_entry: Voce di indice da verificare
            filters: Filtri da applicare
            
        Returns:
            True se la voce corrisponde ai filtri, False altrimenti
        """
        for key, value in filters.items():
            if key not in index_entry:
                return False
            
            if key == 'timestamp':
                # Gestione filtri temporali
                if isinstance(value, dict):
                    if 'start' in value and index_entry[key] < value['start']:
                        return False
                    if 'end' in value and index_entry[key] > value['end']:
                        return False
                else:
                    return False
            elif isinstance(value, (list, tuple)):
                # Filtro a scelta multipla
                if index_entry[key] not in value:
                    return False
            elif isinstance(value, dict) and '$regex' in value:
                # Filtro regex
                import re
                pattern = re.compile(value['$regex'])
                if not pattern.search(str(index_entry[key])):
                    return False
            else:
                # Filtro esatto
                if index_entry[key] != value:
                    return False
        
        return True
    
    def get_unsaid_statistics(self) -> Dict[str, Any]:
        """
        Calcola statistiche sull'archivio del non-detto.
        
        Returns:
            Dizionario con le statistiche
        """
        if not self.index:
            return {"message": "No items in the vault"}
        
        # Statistiche generali
        stats = {
            "total_items": len(self.index),
            "content_types": {},
            "rejection_reasons": {},
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "temporal_distribution": {}
        }
        
        # Analizza le voci dell'indice
        for item_id, index_entry in self.index.items():
            # Distribuzione per tipo di contenuto
            content_type = index_entry["content_type"]
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            # Distribuzione per motivo di rifiuto
            rejection_reason = index_entry.get("rejection_reason", "unknown")
            stats["rejection_reasons"][rejection_reason] = stats["rejection_reasons"].get(rejection_reason, 0) + 1
            
            # Distribuzione della confidenza
            confidence = index_entry["confidence"]
            if confidence < 0.33:
                stats["confidence_distribution"]["low"] += 1
            elif confidence < 0.66:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["high"] += 1
            
            # Distribuzione temporale (per mese)
            timestamp = index_entry["timestamp"]
            if isinstance(timestamp, datetime):
                month_key = timestamp.strftime("%Y-%m")
                stats["temporal_distribution"][month_key] = stats["temporal_distribution"].get(month_key, 0) + 1
        
        return stats
    
    def export_items(self, item_ids: List[str], export_path: str, 
                    format: str = 'json') -> bool:
        """
        Esporta elementi dell'archivio in un file.
        
        Args:
            item_ids: Lista di ID degli elementi da esportare
            export_path: Percorso del file di esportazione
            format: Formato di esportazione ('json', 'csv', 'pickle')
            
        Returns:
            True se l'esportazione è riuscita, False altrimenti
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Recupera gli elementi
            items = []
            for item_id in item_ids:
                item = self.get_unsaid_item(item_id)
                if item:
                    # Converti l'oggetto in un dizionario per la serializzazione
                    item_dict = asdict(item)
                    item_dict['timestamp'] = item.timestamp.isoformat()  # Converti datetime in stringa
                    items.append(item_dict)
            
            if not items:
                return False
            
            # Esporta nel formato specificato
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(items, f)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(items)
                df.to_csv(export_path, index=False)
            else:
                print(f"Formato non supportato: {format}")
                return False
            
            return True
        except Exception as e:
            print(f"Errore durante l'esportazione: {e}")
            return False