"""Analysis package for Football prediction system.

Provide contextual validation, lineup impact analysis, and expected value calculations.
"""

from .lineup_impact import LineupImpactModule, LineupImpact
from .fair_odds_engine import FairOddsEngine, ValuePick, EVAnalysisResult
from .contextual_validation import ContextualValidationPipeline

__all__ = [
    'LineupImpactModule',
    'LineupImpact',
    'FairOddsEngine',
    'ValuePick',
    'EVAnalysisResult',
    'ContextualValidationPipeline'
]
