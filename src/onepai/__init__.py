# -----------------------------------------------------------------------------
# File: onepai/__init__.py
# -----------------------------------------------------------------------------

"""
ONEPAI - Il Tesoro dell'AI
===========================

L'intelligenza dell'invisibile. Un framework per mappare, analizzare e
comprendere le dinamiche nascoste delle reti neurali.

"""

from . import core
from . import introspection
from . import visualization
from . import utils
from . import reader
from . import model
from . import sae

__version__ = "0.4.0-alpha"
__author__ = "Brainverse <francesco.bulla@brainverse.it>"
__all__ = [
    "core", "introspection", "visualization", "utils", "reader", "model", "sae",
    "__version__", "__author__",
]