"""
Spazio del non-pensato - Componente per mappare e analizzare i concetti
e le idee che non sono stati considerati dall'IA durante il suo processo
decisionale.
"""

import json
import pickle
import hashlib
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class UnthoughtConcept:
    """Rappresenta un concetto non considerato dall'IA"""
    id: str
    timestamp: datetime
    concept_type: str  # 'entity', 'relation', 'attribute', 'action', etc.
    concept_description: str
    semantic_vector: Optional[np.ndarray] = None
    context: Dict[str, Any] = None
    related_concepts: List[str] = None  # ID di concetti correlati
    proximity_to_thought: float = 0.0  # Quanto vicino era ad essere considerato (0-1)
    reason_for_not_thinking: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.related_concepts is None:
            self.related_concepts = []

class UnthoughtSpace:
    """
    Sistema per mappare e analizzare i concetti e le idee che non sono stati
    considerati dall'IA durante il suo processo decisionale.
    """
    
    def __init__(self, storage_path: str = "data/treasures/unthought_space",
                 vector_dim: int = 100):
        """
        Inizializza lo spazio del non-pensato.
        
        Args:
            storage_path: Percorso dove salvare i dati dello spazio
            vector_dim: Dimensione dei vettori semantici
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.storage_path / "index.pkl"
        self.concepts_path = self.storage_path / "concepts"
        self.concepts_path.mkdir(exist_ok=True)
        
        self.index = self._load_index()
        self.vector_dim = vector_dim
        
        # Inizializza il vectorizer per il calcolo delle similarità testuali
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fitted_vectorizer = False
        
        # Costruisci il grafo dei concetti
        self.concept_graph = self._build_concept_graph()
    
    def _load_index(self) -> Dict[str, Dict]:
        """
        Carica l'indice dei concetti non-pensati dal disco.
        
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
        """Salva l'indice dei concetti non-pensati su disco."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"Errore nel salvataggio dell'indice: {e}")
    
    def _build_concept_graph(self) -> Dict[str, List[str]]:
        """
        Costruisce un grafo delle relazioni tra concetti non-pensati.
        
        Returns:
            Dizionario che mappa ID di concetti a liste di ID di concetti correlati
        """
        graph = {}
        
        for concept_id, concept_data in self.index.items():
            graph[concept_id] = concept_data.get("related_concepts", [])
        
        return graph
    
    def add_unthought_concept(self, concept_type: str, concept_description: str,
                             context: Dict[str, Any], 
                             proximity_to_thought: float = 0.0,
                             reason_for_not_thinking: Optional[str] = None,
                             semantic_vector: Optional[np.ndarray] = None,
                             related_concepts: Optional[List[str]] = None) -> str:
        """
        Aggiunge un nuovo concetto non-pensato.
        
        Args:
            concept_type: Tipo di concetto
            concept_description: Descrizione testuale del concetto
            context: Contesto in cui il concetto avrebbe potuto essere considerato
            proximity_to_thought: Quanto vicino era ad essere considerato (0-1)
            reason_for_not_thinking: Motivo per cui non è stato considerato
            semantic_vector: Vettore semantico opzionale
            related_concepts: Lista di ID di concetti correlati
            
        Returns:
            ID univoco del concetto aggiunto
        """
        # Genera un ID univoco
        concept_id = str(uuid.uuid4())
        
        # Genera un vettore semantico se non fornito
        if semantic_vector is None:
            semantic_vector = self._generate_semantic_vector(concept_description)
        
        # Crea l'oggetto concetto
        concept = UnthoughtConcept(
            id=concept_id,
            timestamp=datetime.now(),
            concept_type=concept_type,
            concept_description=concept_description,
            semantic_vector=semantic_vector,
            context=context,
            related_concepts=related_concepts or [],
            proximity_to_thought=proximity_to_thought,
            reason_for_not_thinking=reason_for_not_thinking
        )
        
        # Salva il concetto su disco
        self._save_concept(concept)
        
        # Aggiorna l'indice
        self.index[concept_id] = {
            "timestamp": concept.timestamp,
            "concept_type": concept_type,
            "concept_description": concept_description,
            "context": context,
            "proximity_to_thought": proximity_to_thought,
            "reason_for_not_thinking": reason_for_not_thinking,
            "related_concepts": related_concepts or [],
            "file_path": self._get_concept_path(concept_id)
        }
        
        # Aggiorna il grafo dei concetti
        self.concept_graph[concept_id] = related_concepts or []
        
        # Aggiorna il vectorizer se necessario
        if not self.fitted_vectorizer or len(self.index) % 10 == 0:
            self._update_vectorizer()
        
        # Salva l'indice aggiornato
        self._save_index()
        
        return concept_id
    
    def _save_concept(self, concept: UnthoughtConcept):
        """
        Salva i dati di un concetto su disco.
        
        Args:
            concept: Concetto da salvare
        """
        file_path = self._get_concept_path(concept.id)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(concept, f)
        except Exception as e:
            print(f"Errore nel salvataggio del concetto {concept.id}: {e}")
    
    def _get_concept_path(self, concept_id: str) -> Path:
        """
        Ottiene il percorso del file per un concetto dato il suo ID.
        
        Args:
            concept_id: ID del concetto
            
        Returns:
            Percorso del file
        """
        return self.concepts_path / f"{concept_id}.dat"
    
    def _generate_semantic_vector(self, text: str) -> np.ndarray:
        """
        Genera un vettore semantico per un testo dato.
        
        Args:
            text: Testo di cui generare il vettore
            
        Returns:
            Vettore semantico
        """
        # Se abbiamo abbastanza concetti, usa il vectorizer TF-IDF
        if len(self.index) > 5:
            try:
                if not self.fitted_vectorizer:
                    self._update_vectorizer()
                
                # Trasforma il testo in un vettore TF-IDF
                tfidf_vector = self.vectorizer.transform([text])
                
                # Converti in un vettore denso della dimensione richiesta
                dense_vector = tfidf_vector.toarray().flatten()
                
                # Ridimensiona o espandi il vettore alla dimensione richiesta
                if len(dense_vector) >= self.vector_dim:
                    return dense_vector[:self.vector_dim]
                else:
                    # Espandi il vettore con zeri
                    padded_vector = np.zeros(self.vector_dim)
                    padded_vector[:len(dense_vector)] = dense_vector
                    return padded_vector
            except Exception as e:
                print(f"Errore nella generazione del vettore semantico: {e}")
        
        # Fallback: genera un vettore casuale
        return np.random.rand(self.vector_dim)
    
    def _update_vectorizer(self):
        """Aggiorna il vectorizer TF-IDF con i concetti esistenti."""
        if len(self.index) < 2:
            return
        
        try:
            # Raccogli tutte le descrizioni dei concetti
            descriptions = [data["concept_description"] for data in self.index.values()]
            
            # Adatta il vectorizer
            self.vectorizer.fit(descriptions)
            self.fitted_vectorizer = True
        except Exception as e:
            print(f"Errore nell'aggiornamento del vectorizer: {e}")
    
    def get_unthought_concept(self, concept_id: str) -> Optional[UnthoughtConcept]:
        """
        Recupera un concetto dallo spazio dato il suo ID.
        
        Args:
            concept_id: ID del concetto
            
        Returns:
            Il concetto richiesto o None se non trovato
        """
        if concept_id not in self.index:
            return None
        
        index_entry = self.index[concept_id]
        file_path = Path(index_entry["file_path"])
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Errore nel caricamento del concetto {concept_id}: {e}")
            return None
    
    def query_unthought_concepts(self, filters: Dict[str, Any] = None, 
                               limit: Optional[int] = None) -> List[UnthoughtConcept]:
        """
        Interroga lo spazio per concetti che corrispondono a filtri specifici.
        
        Args:
            filters: Dizionario di filtri da applicare
            limit: Numero massimo di risultati
            
        Returns:
            Lista di concetti che corrispondono ai filtri
        """
        results = []
        
        for concept_id, index_entry in self.index.items():
            # Applica i filtri
            if filters and not self._concept_matches_filters(index_entry, filters):
                continue
            
            # Carica il concetto completo
            concept = self.get_unthought_concept(concept_id)
            if concept:
                results.append(concept)
                
                # Controlla il limite
                if limit is not None and len(results) >= limit:
                    break
        
        # Ordina per proximity_to_thought (decrescente)
        results.sort(key=lambda x: x.proximity_to_thought, reverse=True)
        
        return results
    
    def _concept_matches_filters(self, index_entry: Dict[str, Any], 
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
            elif key == 'proximity_to_thought':
                # Gestione filtri numerici con operatori
                if isinstance(value, dict):
                    if '$gt' in value and index_entry[key] <= value['$gt']:
                        return False
                    if '$lt' in value and index_entry[key] >= value['$lt']:
                        return False
                    if '$gte' in value and index_entry[key] < value['$gte']:
                        return False
                    if '$lte' in value and index_entry[key] > value['$lte']:
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
    
    def find_similar_concepts(self, concept_id: str, 
                            threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Trova concetti simili a un dato concetto.
        
        Args:
            concept_id: ID del concetto di partenza
            threshold: Soglia di similarità
            
        Returns:
            Lista di tuple (ID_concetto, similarità) ordinate per similarità decrescente
        """
        if concept_id not in self.index:
            return []
        
        # Recupera il concetto di riferimento
        reference_concept = self.get_unthought_concept(concept_id)
        if not reference_concept or reference_concept.semantic_vector is None:
            return []
        
        similarities = []
        
        # Calcola la similarità con tutti gli altri concetti
        for other_id, index_entry in self.index.items():
            if other_id == concept_id:
                continue
            
            other_concept = self.get_unthought_concept(other_id)
            if not other_concept or other_concept.semantic_vector is None:
                continue
            
            # Calcola la similarità coseno tra i vettori semantici
            similarity = cosine_similarity(
                reference_concept.semantic_vector.reshape(1, -1),
                other_concept.semantic_vector.reshape(1, -1)
            )[0, 0]
            
            if similarity >= threshold:
                similarities.append((other_id, similarity))
        
        # Ordina per similarità decrescente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def get_concept_clusters(self, threshold: float = 0.6) -> List[List[str]]:
        """
        Identifica cluster di concetti simili tra loro.
        
        Args:
            threshold: Soglia di similarità per considerare due concetti correlati
            
        Returns:
            Lista di cluster, dove ogni cluster è una lista di ID di concetti
        """
        if len(self.index) < 2:
            return []
        
        # Costruisci una matrice di similarità
        concept_ids = list(self.index.keys())
        similarity_matrix = np.zeros((len(concept_ids), len(concept_ids)))
        
        for i, id1 in enumerate(concept_ids):
            concept1 = self.get_unthought_concept(id1)
            if not concept1 or concept1.semantic_vector is None:
                continue
            
            for j, id2 in enumerate(concept_ids[i+1:], i+1):
                concept2 = self.get_unthought_concept(id2)
                if not concept2 or concept2.semantic_vector is None:
                    continue
                
                # Calcola la similarità coseno
                similarity = cosine_similarity(
                    concept1.semantic_vector.reshape(1, -1),
                    concept2.semantic_vector.reshape(1, -1)
                )[0, 0]
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Applica un algoritmo di clustering semplice (basato sulla soglia)
        clusters = []
        visited = set()
        
        for i, concept_id in enumerate(concept_ids):
            if concept_id in visited:
                continue
            
            # Avvia un nuovo cluster
            cluster = [concept_id]
            visited.add(concept_id)
            
            # Trova tutti i concetti simili non visitati
            self._find_similar_concepts(i, similarity_matrix, concept_ids, cluster, visited, threshold)
            
            # Aggiungi il cluster se ha più di un concetto
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _find_similar_concepts(self, index: int, similarity_matrix: np.ndarray,
                              concept_ids: List[str], cluster: List[str],
                              visited: Set[str], threshold: float):
        """
        Trova ricorsivamente concetti simili a un dato concetto.
        
        Args:
            index: Indice del concetto di partenza
            similarity_matrix: Matrice di similarità
            concept_ids: Lista di ID dei concetti
            cluster: Cluster in costruzione
            visited: Insieme dei concetti già visitati
            threshold: Soglia di similarità
        """
        for j, other_id in enumerate(concept_ids):
            if other_id in visited:
                continue
            
            # Controlla se i concetti sono sufficientemente simili
            if similarity_matrix[index, j] >= threshold:
                cluster.append(other_id)
                visited.add(other_id)
                
                # Continua la ricerca ricorsiva
                self._find_similar_concepts(j, similarity_matrix, concept_ids, cluster, visited, threshold)
    
    def get_unthought_statistics(self) -> Dict[str, Any]:
        """
        Calcola statistiche sullo spazio del non-pensato.
        
        Returns:
            Dizionario con le statistiche
        """
        if not self.index:
            return {"message": "No concepts in the unthought space"}
        
        # Statistiche generali
        stats = {
            "total_concepts": len(self.index),
            "concept_types": {},
            "reasons_for_not_thinking": {},
            "proximity_distribution": {"low": 0, "medium": 0, "high": 0},
            "temporal_distribution": {},
            "avg_related_concepts": 0,
            "graph_density": 0.0
        }
        
        total_relations = 0
        
        # Analizza le voci dell'indice
        for concept_id, index_entry in self.index.items():
            # Distribuzione per tipo di concetto
            concept_type = index_entry["concept_type"]
            stats["concept_types"][concept_type] = stats["concept_types"].get(concept_type, 0) + 1
            
            # Distribuzione per motivo di non-pensiero
            reason = index_entry.get("reason_for_not_thinking", "unknown")
            stats["reasons_for_not_thinking"][reason] = stats["reasons_for_not_thinking"].get(reason, 0) + 1
            
            # Distribuzione della vicinanza al pensiero
            proximity = index_entry["proximity_to_thought"]
            if proximity < 0.33:
                stats["proximity_distribution"]["low"] += 1
            elif proximity < 0.66:
                stats["proximity_distribution"]["medium"] += 1
            else:
                stats["proximity_distribution"]["high"] += 1
            
            # Distribuzione temporale (per mese)
            timestamp = index_entry["timestamp"]
            if isinstance(timestamp, datetime):
                month_key = timestamp.strftime("%Y-%m")
                stats["temporal_distribution"][month_key] = stats["temporal_distribution"].get(month_key, 0) + 1
            
            # Conta le relazioni
            relations_count = len(index_entry.get("related_concepts", []))
            total_relations += relations_count
        
        # Calcola la media delle relazioni per concetto
        if self.index:
            stats["avg_related_concepts"] = total_relations / len(self.index)
        
        # Calcola la densità del grafo
        if len(self.index) > 1:
            max_possible_edges = len(self.index) * (len(self.index) - 1)
            stats["graph_density"] = total_relations / max_possible_edges
        
        return stats
    
    def find_concept_gaps(self, context: Dict[str, Any]) -> List[str]:
        """
        Identifica concetti mancanti in un dato contesto.
        
        Args:
            context: Contesto in cui cercare concetti mancanti
            
        Returns:
            Lista di ID di concetti che sarebbero stati rilevanti nel contesto
        """
        # Estrai parole chiave dal contesto
        context_text = " ".join(str(v) for v in context.values())
        
        # Trova concetti simili al contesto
        similar_concepts = []
        
        for concept_id, index_entry in self.index.items():
            concept = self.get_unthought_concept(concept_id)
            if not concept:
                continue
            
            # Calcola la similarità tra la descrizione del concetto e il contesto
            concept_text = concept.concept_description
            
            try:
                if not self.fitted_vectorizer:
                    self._update_vectorizer()
                
                # Trasforma entrambi i testi in vettori TF-IDF
                context_vector = self.vectorizer.transform([context_text])
                concept_vector = self.vectorizer.transform([concept_text])
                
                # Calcola la similarità coseno
                similarity = cosine_similarity(context_vector, concept_vector)[0, 0]
                
                if similarity > 0.3:  # Soglia arbitraria
                    similar_concepts.append((concept_id, similarity))
            except Exception as e:
                print(f"Errore nel calcolo della similarità per il concetto {concept_id}: {e}")
        
        # Ordina per similarità decrescente
        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Restituisci gli ID dei concetti più simili
        return [concept_id for concept_id, _ in similar_concepts[:5]]
    
    def export_concepts(self, concept_ids: List[str], export_path: str, 
                       format: str = 'json') -> bool:
        """
        Esporta concetti dello spazio in un file.
        
        Args:
            concept_ids: Lista di ID dei concetti da esportare
            export_path: Percorso del file di esportazione
            format: Formato di esportazione ('json', 'csv', 'pickle')
            
        Returns:
            True se l'esportazione è riuscita, False altrimenti
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Recupera i concetti
            concepts = []
            for concept_id in concept_ids:
                concept = self.get_unthought_concept(concept_id)
                if concept:
                    # Converti l'oggetto in un dizionario per la serializzazione
                    concept_dict = asdict(concept)
                    concept_dict['timestamp'] = concept.timestamp.isoformat()  # Converti datetime in stringa
                    if concept.semantic_vector is not None:
                        concept_dict['semantic_vector'] = concept.semantic_vector.tolist()  # Converti numpy array in lista
                    concepts.append(concept_dict)
            
            if not concepts:
                return False
            
            # Esporta nel formato specificato
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(concepts, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(concepts, f)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(concepts)
                df.to_csv(export_path, index=False)
            else:
                print(f"Formato non supportato: {format}")
                return False
            
            return True
        except Exception as e:
            print(f"Errore durante l'esportazione: {e}")
            return False