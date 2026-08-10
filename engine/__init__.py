"""Engine package for Football prediction system.

Provides data fetching, processing, and ML-based prediction capabilities.
"""

from .api_client import FootballDataClient
from .processor import DataProcessor
from .models import CatBoostPredictor, LSTMMomentumFilter

__all__ = [
    'FootballDataClient',
    'DataProcessor',
    'CatBoostPredictor',
    'LSTMMomentumFilter'
]
