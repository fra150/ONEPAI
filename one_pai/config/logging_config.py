"""
ONEPAI Logging Configuration
============================

Sistema di logging avanzato per ONEPAI - Il Tesoro dell'AI
Traccia tutto ciò che accade nel regno dell'invisibile.

Author: Francesco Bulla (Brainverse)
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LogConfig:
    """Configurazione logging ONEPAI"""
    level: str = "INFO"
    format_type: str = "shadow"  # shadow, void, silence, treasure
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_path: str = "logs/onepai.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_shadow_logs: bool = True    # Log specifici per shadow mapping
    enable_silence_logs: bool = True   # Log specifici per silence tracing
    enable_void_logs: bool = True      # Log specifici per void analysis
    enable_treasure_logs: bool = True  # Log specifici per treasure operations

class OnepaiFormatter(logging.Formatter):
    """
    Formatter personalizzato per ONEPAI
    Aggiunge simboli e colori per diversi tipi di log
    """
    
    # Simboli per diversi tipi di log
    SYMBOLS = {
        'shadow': '🌑',
        'silence': '🔇', 
        'void': '🕳️',
        'treasure': '💎',
        'observer': '👁️',
        'neural': '🧠',
        'memory': '💭',
        'analysis': '📊',
        'crypto': '🔐',
        'api': '🌐',
        'error': '🚨',
        'warning': '⚠️',
        'info': '💫',
        'debug': '🔍'
    }
    
    # Colori ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[37m',       # White
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'SHADOW': '\033[90m',     # Dark Gray
        'SILENCE': '\033[94m',    # Blue
        'VOID': '\033[95m',       # Purple
        'TREASURE': '\033[93m',   # Yellow
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, format_type: str = "shadow", use_colors: bool = True):
        self.format_type = format_type
        self.use_colors = use_colors
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatta il record di log con simboli e colori ONEPAI
        """
        # Determina il tipo di log dal nome del modulo
        log_type = self._get_log_type(record)
        symbol = self.SYMBOLS.get(log_type, self.SYMBOLS.get(record.levelname.lower(), '📝'))
        
        # Timestamp formattato
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Informazioni extra se presenti
        extra_info = ""
        if hasattr(record, 'shadow_id'):
            extra_info += f" [Shadow:{record.shadow_id}]"
        if hasattr(record, 'treasure_id'):
            extra_info += f" [Treasure:{record.treasure_id}]"
        if hasattr(record, 'void_depth'):
            extra_info += f" [Void:{record.void_depth}]"
        if hasattr(record, 'silence_type'):
            extra_info += f" [Silence:{record.silence_type}]"
        
        # Formato base
        base_format = f"{symbol} {timestamp} [{record.levelname:8}] {record.name:20} {extra_info} | {record.getMessage()}"
        
        # Aggiunge colori se abilitati e se siamo in un terminal
        if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
            base_format = f"{color}{base_format}{self.COLORS['RESET']}"
        
        # Aggiunge stack trace per errori
        if record.exc_info:
            base_format += "\n" + self.formatException(record.exc_info)
        
        return base_format
    
    def _get_log_type(self, record: logging.LogRecord) -> str:
        """Determina il tipo di log dal nome del modulo"""
        module_name = record.name.lower()
        
        if 'shadow' in module_name:
            return 'shadow'
        elif 'silence' in module_name:
            return 'silence'
        elif 'void' in module_name:
            return 'void'
        elif 'treasure' in module_name:
            return 'treasure'
        elif 'observer' in module_name:
            return 'observer'
        elif 'neural' in module_name:
            return 'neural'
        elif 'memory' in module_name:
            return 'memory'
        elif 'analysis' in module_name:
            return 'analysis'
        elif 'crypto' in module_name:
            return 'crypto'
        elif 'api' in module_name:
            return 'api'
        else:
            return record.levelname.lower()

class OnepaiLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter personalizzato per aggiungere contesto ONEPAI ai log
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Aggiunge informazioni di contesto al messaggio"""
        return msg, kwargs
    
    def shadow(self, msg: str, shadow_id: Optional[str] = None, **kwargs):
        """Log specifico per operazioni shadow"""
        extra = kwargs.get('extra', {})
        if shadow_id:
            extra['shadow_id'] = shadow_id
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def silence(self, msg: str, silence_type: Optional[str] = None, **kwargs):
        """Log specifico per operazioni silence"""
        extra = kwargs.get('extra', {})
        if silence_type:
            extra['silence_type'] = silence_type
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def void(self, msg: str, void_depth: Optional[int] = None, **kwargs):
        """Log specifico per operazioni void"""
        extra = kwargs.get('extra', {})
        if void_depth:
            extra['void_depth'] = void_depth
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def treasure(self, msg: str, treasure_id: Optional[str] = None, **kwargs):
        """Log specifico per operazioni treasure"""
        extra = kwargs.get('extra', {})
        if treasure_id:
            extra['treasure_id'] = treasure_id
        kwargs['extra'] = extra
        self.info(msg, **kwargs)

def setup_logging(config: Optional[LogConfig] = None) -> OnepaiLoggerAdapter:
    """
    Configura il sistema di logging ONEPAI
    
    Args:
        config: Configurazione logging (usa default se None)
        
    Returns:
        OnepaiLoggerAdapter: Logger principale configurato
    """
    if config is None:
        config = LogConfig()
    
    # Logger principale
    logger = logging.getLogger("onepai")
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Rimuove handler esistenti
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formatter personalizzato
    formatter = OnepaiFormatter(
        format_type=config.format_type,
        use_colors=config.log_to_console
    )
    
    # Handler per console
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler per file
    if config.log_to_file:
        # Crea directory log se non esiste
        Path(config.log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        
        # Formatter senza colori per file
        file_formatter = OnepaiFormatter(
            format_type=config.format_type,
            use_colors=False
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Adapter personalizzato
    adapter = OnepaiLoggerAdapter(logger, {})
    
    # Log di inizializzazione
    adapter.info("🔥 ONEPAI Logging System Initialized")
    adapter.info(f"📊 Log Level: {config.level}")
    adapter.info(f"📝 Format Type: {config.format_type}")
    adapter.info(f"💾 Log to File: {config.log_to_file}")
    adapter.info(f"🖥️ Log to Console: {config.log_to_console}")
    
    return adapter

def get_logger(name: str = "onepai") -> OnepaiLoggerAdapter:
    """
    Ottiene un logger ONEPAI per un modulo specifico  
    Args:
        name: Nome del modulo/componente      
    Returns:
        OnepaiLoggerAdapter: Logger configurato
    """
    logger = logging.getLogger(name)
    return OnepaiLoggerAdapter(logger, {})

# Logger globale predefinito
_default_logger: Optional[OnepaiLoggerAdapter] = None

def get_default_logger() -> OnepaiLoggerAdapter:
    """Ottiene il logger globale predefinito"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger