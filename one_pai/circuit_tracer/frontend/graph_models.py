# -----------------------------------------------------------------------------
# File: circuit_tracer/models.py
# -----------------------------------------------------------------------------

"""
Modelli per il circuit tracer.

Questo modulo fornisce:
- ReplacementModel per l'analisi di attribuzione
- Wrapper per modelli Hugging Face
- Gestione dei transcoder
- Utility per il caricamento e configurazione modelli
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReplacementModel(nn.Module):
    """
    Modello wrapper per l'analisi di attribuzione con transcoder.
    
    Questo modello permette di:
    - Caricare modelli da Hugging Face
    - Integrare transcoder per l'analisi
    - Gestire offloading per memoria limitata
    - Supportare lazy loading
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        transcoder_set: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "auto",
        offload: Optional[str] = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.transcoder_set = transcoder_set
        self.dtype = dtype
        self.device = device if device != "auto" else self._get_best_device()
        self.offload = offload
        self.lazy_encoder = lazy_encoder
        self.lazy_decoder = lazy_decoder
        
        # Placeholder per il modello e transcoder
        self.model = None
        self.transcoder = None
        self.tokenizer = None
        
        logger.info(f"Inizializzato ReplacementModel con transcoder_set: {transcoder_set}")
        
        # Carica il modello se specificato
        if model_name or transcoder_set:
            self._load_model()
    
    def _get_best_device(self) -> str:
        """Determina il miglior device disponibile."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Carica il modello e i transcoder."""
        try:
            # Simulazione del caricamento del modello
            logger.info(f"Caricamento modello: {self.model_name or 'default'}")
            logger.info(f"Caricamento transcoder set: {self.transcoder_set}")
            
            # Per ora, creiamo un modello placeholder
            # In una implementazione reale, qui caricheresti da Hugging Face
            self.model = self._create_placeholder_model()
            self.transcoder = self._create_placeholder_transcoder()
            self.tokenizer = self._create_placeholder_tokenizer()
            
            # Applica configurazioni
            if self.offload:
                logger.info(f"Applicando offload: {self.offload}")
                self._apply_offload()
            
            if self.lazy_encoder:
                logger.info("Abilitando lazy loading per encoder")
            
            if self.lazy_decoder:
                logger.info("Abilitando lazy loading per decoder")
                
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            raise
    
    def _create_placeholder_model(self) -> nn.Module:
        """Crea un modello placeholder per testing."""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50257)  # Vocab size tipico
        )
    
    def _create_placeholder_transcoder(self) -> Dict[str, Any]:
        """Crea transcoder placeholder."""
        return {
            "encoder": nn.Linear(768, 512),
            "decoder": nn.Linear(512, 768),
            "config": {"hidden_size": 512, "num_layers": 12}
        }
    
    def _create_placeholder_tokenizer(self) -> Dict[str, Any]:
        """Crea tokenizer placeholder."""
        return {
            "vocab_size": 50257,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "encode": lambda text: [1, 2, 3, 4, 5],  # Placeholder encoding
            "decode": lambda ids: "placeholder text"
        }
    
    def _apply_offload(self):
        """Applica offloading del modello."""
        if self.offload == "cpu":
            logger.info("Offloading parametri su CPU")
        elif self.offload == "disk":
            logger.info("Offloading parametri su disco")
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass del modello."""
        if self.model is None:
            raise RuntimeError("Modello non caricato. Chiamare _load_model() prima.")
        
        # Converti input_ids in embeddings (placeholder)
        batch_size, seq_len = input_ids.shape
        embeddings = torch.randn(batch_size, seq_len, 768, dtype=self.dtype, device=self.device)
        
        # Forward pass attraverso il modello
        output = self.model(embeddings.view(batch_size * seq_len, -1))
        
        return output.view(batch_size, seq_len, -1)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Codifica testo in token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato")
        
        # Placeholder encoding
        token_ids = self.tokenizer["encode"](text)
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decodifica token IDs in testo."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato")
        
        # Placeholder decoding
        return self.tokenizer["decode"](token_ids.tolist())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello."""
        return {
            "model_name": self.model_name,
            "transcoder_set": self.transcoder_set,
            "dtype": str(self.dtype),
            "device": self.device,
            "offload": self.offload,
            "lazy_encoder": self.lazy_encoder,
            "lazy_decoder": self.lazy_decoder,
            "parameters": sum(p.numel() for p in self.parameters()) if self.model else 0
        }
    
    def save_model(self, path: str):
        """Salva il modello su disco."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.get_model_info()
        }, save_path)
        
        logger.info(f"Modello salvato in: {save_path}")
    
    @classmethod
    def load_model(cls, path: str) -> "ReplacementModel":
        """Carica un modello da disco."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Modello caricato da: {path}")
        return model
    
    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoder: Dict[str, Any],
        dtype: torch.dtype = torch.float32,
        **kwargs
    ) -> "ReplacementModel":
        """
        Crea ReplacementModel da modello pre-addestrato e transcoder.
        
        Args:
            model_name: Nome del modello Hugging Face
            transcoder: Transcoder caricato
            dtype: Tipo di dato per i parametri
            **kwargs: Parametri aggiuntivi
        
        Returns:
            ReplacementModel configurato
        """
        
        logger.info(f"Creazione ReplacementModel da {model_name} con transcoder")
        
        # Crea istanza con transcoder
        model = cls(
            model_name=model_name,
            dtype=dtype,
            **kwargs
        )
        
        # Assegna transcoder caricato
        model.transcoder = transcoder
        
        # Configura attributi aggiuntivi necessari per attribution
        from .graph import HookedTransformerConfig
        model.cfg = HookedTransformerConfig(
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257
        )
        model.scan = getattr(model, 'scan', 'default')
        
        # Aggiungi hook methods necessari per attribution
        model.feature_input_hook = lambda x: x  # Placeholder
        model.feature_output_hook = lambda x: x  # Placeholder
        
        # Context manager per hooks
        from contextlib import contextmanager
        
        @contextmanager
        def hooks(fwd_hooks=None, bwd_hooks=None):
            """Context manager per gestire hooks."""
            try:
                yield
            finally:
                pass
        
        model.hooks = hooks
        
        logger.info("ReplacementModel configurato con successo")
        return model


def create_model(
    transcoder_set: str,
    model_name: Optional[str] = None,
    **kwargs
) -> ReplacementModel:
    """
    Factory function per creare un ReplacementModel.
    
    Args:
        transcoder_set: Repository Hugging Face con i transcoder
        model_name: Nome del modello (opzionale)
        **kwargs: Parametri aggiuntivi per ReplacementModel
    
    Returns:
        ReplacementModel configurato
    """
    return ReplacementModel(
        model_name=model_name,
        transcoder_set=transcoder_set,
        **kwargs
    )