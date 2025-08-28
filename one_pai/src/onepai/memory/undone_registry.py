"""
Registro del non-agito - Componente per tracciare e analizzare le azioni
che sono state considerate ma non eseguite dall'IA.
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
from enum import Enum

class ActionType(Enum):
    """Tipi di azioni che possono essere registrate"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    DECISION = "decision"
    API_CALL = "api_call"
    FUNCTION_CALL = "function_call"
    SEARCH = "search"
    OTHER = "other"

@dataclass
class UndoneAction:
    """Rappresenta un'azione considerata ma non eseguita"""
    id: str
    timestamp: datetime
    action_type: ActionType
    action_description: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    confidence: float  # Livello di confidenza che questa azione fosse appropriata
    rejection_reason: Optional[str] = None
    expected_outcome: Optional[Dict[str, Any]] = None
    alternative_actions: List[str] = None  # ID di azioni alternative
    
    def __post_init__(self):
        if self.alternative_actions is None:
            self.alternative_actions = []

class UndoneRegistry:
    """
    Sistema per registrare e analizzare le azioni che sono state considerate
    ma non eseguite dall'IA durante il suo funzionamento.
    """
    
    def __init__(self, storage_path: str = "data/treasures/undone_registry"):
        """
        Inizializza il registro del non-agito.
        
        Args:
            storage_path: Percorso dove salvare i dati del registro
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.storage_path / "index.pkl"
        self.actions_path = self.storage_path / "actions"
        self.actions_path.mkdir(exist_ok=True)
        
        self.index = self._load_index()
        self.action_graph = self._build_action_graph()
    
    def _load_index(self) -> Dict[str, Dict]:
        """
        Carica l'indice delle azioni non-eseguite dal disco.
        
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
        """Salva l'indice delle azioni non-eseguite su disco."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"Errore nel salvataggio dell'indice: {e}")
    
    def _build_action_graph(self) -> Dict[str, List[str]]:
        """
        Costruisce un grafo delle relazioni tra azioni non-eseguite.
        
        Returns:
            Dizionario che mappa ID di azioni a liste di ID di azioni correlate
        """
        graph = {}
        
        for action_id, action_data in self.index.items():
            graph[action_id] = action_data.get("alternative_actions", [])
        
        return graph
    
    def register_undone_action(self, action_type: Union[ActionType, str], 
                              action_description: str, parameters: Dict[str, Any],
                              context: Dict[str, Any], confidence: float,
                              rejection_reason: Optional[str] = None,
                              expected_outcome: Optional[Dict[str, Any]] = None,
                              alternative_actions: Optional[List[str]] = None) -> str:
        """
        Registra una nuova azione non-eseguita.
        
        Args:
            action_type: Tipo di azione
            action_description: Descrizione testuale dell'azione
            parameters: Parametri dell'azione
            context: Contesto in cui è stata considerata l'azione
            confidence: Livello di confidenza (0-1)
            rejection_reason: Motivo della non-esecuzione
            expected_outcome: Risultato atteso se l'azione fosse stata eseguita
            alternative_actions: Lista di ID di azioni alternative considerate
            
        Returns:
            ID univoco dell'azione registrata
        """
        # Normalizza action_type se è una stringa
        if isinstance(action_type, str):
            try:
                action_type = ActionType(action_type)
            except ValueError:
                action_type = ActionType.OTHER
        
        # Genera un ID univoco
        action_id = str(uuid.uuid4())
        
        # Crea l'oggetto azione
        action = UndoneAction(
            id=action_id,
            timestamp=datetime.now(),
            action_type=action_type,
            action_description=action_description,
            parameters=parameters,
            context=context,
            confidence=confidence,
            rejection_reason=rejection_reason,
            expected_outcome=expected_outcome,
            alternative_actions=alternative_actions or []
        )
        
        # Salva l'azione su disco
        self._save_action(action)
        
        # Aggiorna l'indice
        self.index[action_id] = {
            "timestamp": action.timestamp,
            "action_type": action.action_type.value,
            "action_description": action_description,
            "parameters_hash": self._compute_parameters_hash(parameters),
            "context": context,
            "confidence": confidence,
            "rejection_reason": rejection_reason,
            "expected_outcome": expected_outcome,
            "alternative_actions": alternative_actions or [],
            "file_path": self._get_action_path(action_id)
        }
        
        # Aggiorna il grafo delle azioni
        self.action_graph[action_id] = alternative_actions or []
        
        # Salva l'indice aggiornato
        self._save_index()
        
        return action_id
    
    def _save_action(self, action: UndoneAction):
        """
        Salva i dati di un'azione su disco.
        
        Args:
            action: Azione da salvare
        """
        file_path = self._get_action_path(action.id)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(action, f)
        except Exception as e:
            print(f"Errore nel salvataggio dell'azione {action.id}: {e}")
    
    def _get_action_path(self, action_id: str) -> Path:
        """
        Ottiene il percorso del file per un'azione dato il suo ID.
        
        Args:
            action_id: ID dell'azione
            
        Returns:
            Percorso del file
        """
        return self.actions_path / f"{action_id}.dat"
    
    def _compute_parameters_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Calcola un hash dei parametri per identificazione univoca.
        
        Args:
            parameters: Parametri di cui calcolare l'hash
            
        Returns:
            Stringa hash
        """
        try:
            # Prova a serializzare in JSON
            params_str = json.dumps(parameters, sort_keys=True)
        except (TypeError, OverflowError):
            # Se non è possibile, usa pickle
            params_bytes = pickle.dumps(parameters)
            params_str = params_bytes.decode('latin1')
        
        return hashlib.md5(params_str.encode('utf-8')).hexdigest()
    
    def get_undone_action(self, action_id: str) -> Optional[UndoneAction]:
        """
        Recupera un'azione dal registro dato il suo ID.
        
        Args:
            action_id: ID dell'azione
            
        Returns:
            L'azione richiesta o None se non trovata
        """
        if action_id not in self.index:
            return None
        
        index_entry = self.index[action_id]
        file_path = Path(index_entry["file_path"])
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Errore nel caricamento dell'azione {action_id}: {e}")
            return None
    
    def query_undone_actions(self, filters: Dict[str, Any] = None, 
                           limit: Optional[int] = None) -> List[UndoneAction]:
        """
        Interroga il registro per azioni che corrispondono a filtri specifici.
        
        Args:
            filters: Dizionario di filtri da applicare
            limit: Numero massimo di risultati
            
        Returns:
            Lista di azioni che corrispondono ai filtri
        """
        results = []
        
        for action_id, index_entry in self.index.items():
            # Applica i filtri
            if filters and not self._action_matches_filters(index_entry, filters):
                continue
            
            # Carica l'azione completa
            action = self.get_undone_action(action_id)
            if action:
                results.append(action)
                
                # Controlla il limite
                if limit is not None and len(results) >= limit:
                    break
        
        # Ordina per timestamp (più recenti prima)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def _action_matches_filters(self, index_entry: Dict[str, Any], 
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
            elif key == 'action_type':
                # Gestione del tipo di azione (enum o stringa)
                if isinstance(value, ActionType):
                    if index_entry[key] != value.value:
                        return False
                elif isinstance(value, str):
                    if index_entry[key] != value:
                        return False
                elif isinstance(value, list):
                    if index_entry[key] not in [v.value if isinstance(v, ActionType) else v for v in value]:
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
    
    def get_action_alternatives(self, action_id: str) -> List[UndoneAction]:
        """
        Recupera le azioni alternative a una data azione.
        
        Args:
            action_id: ID dell'azione di partenza
            
        Returns:
            Lista di azioni alternative
        """
        if action_id not in self.action_graph:
            return []
        
        alternative_ids = self.action_graph[action_id]
        alternatives = []
        
        for alt_id in alternative_ids:
            action = self.get_undone_action(alt_id)
            if action:
                alternatives.append(action)
        
        return alternatives
    
    def get_undone_statistics(self) -> Dict[str, Any]:
        """
        Calcola statistiche sul registro del non-agito.
        
        Returns:
            Dizionario con le statistiche
        """
        if not self.index:
            return {"message": "No actions in the registry"}
        
        # Statistiche generali
        stats = {
            "total_actions": len(self.index),
            "action_types": {},
            "rejection_reasons": {},
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "temporal_distribution": {},
            "avg_alternatives_per_action": 0
        }
        
        total_alternatives = 0
        
        # Analizza le voci dell'indice
        for action_id, index_entry in self.index.items():
            # Distribuzione per tipo di azione
            action_type = index_entry["action_type"]
            stats["action_types"][action_type] = stats["action_types"].get(action_type, 0) + 1
            
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
            
            # Conta le alternative
            alternatives_count = len(index_entry.get("alternative_actions", []))
            total_alternatives += alternatives_count
        
        # Calcola la media delle alternative per azione
        if self.index:
            stats["avg_alternatives_per_action"] = total_alternatives / len(self.index)
        
        return stats
    
    def find_action_patterns(self) -> Dict[str, Any]:
        """
        Identifica pattern nelle azioni non-eseguite.
        
        Returns:
            Dizionario con i pattern identificati
        """
        if not self.index:
            return {"message": "No actions in the registry"}
        
        patterns = {
            "common_parameter_combinations": {},
            "frequent_rejection_contexts": {},
            "action_chains": []
        }
        
        # Analizza le combinazioni di parametri comuni
        param_combinations = {}
        for action_id, index_entry in self.index.items():
            action = self.get_undone_action(action_id)
            if not action:
                continue
            
            # Estrai una firma semplificata dei parametri
            param_signature = tuple(sorted(action.parameters.keys()))
            param_combinations[param_signature] = param_combinations.get(param_signature, 0) + 1
        
        # Trova le combinazioni più comuni
        patterns["common_parameter_combinations"] = sorted(
            param_combinations.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Analizza i contesti di rifiuto frequenti
        context_features = {}
        for action_id, index_entry in self.index.items():
            action = self.get_undone_action(action_id)
            if not action or not action.rejection_reason:
                continue
            
            # Estrai caratteristiche del contesto
            for key, value in action.context.items():
                feature = f"{key}:{str(value)[:50]}"  # Limita la lunghezza del valore
                context_features[feature] = context_features.get(feature, 0) + 1
        
        # Trova le caratteristiche di contesto più comuni
        patterns["frequent_rejection_contexts"] = sorted(
            context_features.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Analizza le catene di azioni (sequenze di azioni alternative)
        action_chains = []
        visited = set()
        
        for action_id in self.action_graph:
            if action_id in visited:
                continue
            
            # Trova tutte le azioni collegate (componente connessa)
            chain = self._find_action_chain(action_id, visited)
            if len(chain) > 1:  # Considera solo catene con più di un'azione
                action_chains.append(chain)
        
        # Ordina le catene per lunghezza (decrescente)
        action_chains.sort(key=len, reverse=True)
        patterns["action_chains"] = action_chains[:5]  # Prendi le 5 catene più lunghe
        
        return patterns
    
    def _find_action_chain(self, start_id: str, visited: set) -> List[str]:
        """
        Trova ricorsivamente una catena di azioni collegate.
        
        Args:
            start_id: ID di inizio della catena
            visited: Insieme di ID già visitati
            
        Returns:
            Lista di ID che formano la catena
        """
        if start_id in visited:
            return []
        
        visited.add(start_id)
        chain = [start_id]
        
        # Esplora le azioni collegate
        for connected_id in self.action_graph.get(start_id, []):
            if connected_id not in visited:
                subchain = self._find_action_chain(connected_id, visited)
                chain.extend(subchain)
        
        # Esplora le azioni che puntano a questa
        for action_id, alternatives in self.action_graph.items():
            if start_id in alternatives and action_id not in visited:
                subchain = self._find_action_chain(action_id, visited)
                chain.extend(subchain)
        
        return chain
    
    def export_actions(self, action_ids: List[str], export_path: str, 
                      format: str = 'json') -> bool:
        """
        Esporta azioni del registro in un file.
        
        Args:
            action_ids: Lista di ID delle azioni da esportare
            export_path: Percorso del file di esportazione
            format: Formato di esportazione ('json', 'csv', 'pickle')
            
        Returns:
            True se l'esportazione è riuscita, False altrimenti
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Recupera le azioni
            actions = []
            for action_id in action_ids:
                action = self.get_undone_action(action_id)
                if action:
                    # Converti l'oggetto in un dizionario per la serializzazione
                    action_dict = asdict(action)
                    action_dict['timestamp'] = action.timestamp.isoformat()  # Converti datetime in stringa
                    action_dict['action_type'] = action.action_type.value  # Converti enum in stringa
                    actions.append(action_dict)
            
            if not actions:
                return False
            
            # Esporta nel formato specificato
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(actions, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(actions, f)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(actions)
                df.to_csv(export_path, index=False)
            else:
                print(f"Formato non supportato: {format}")
                return False
            
            return True
        except Exception as e:
            print(f"Errore durante l'esportazione: {e}")
            return False