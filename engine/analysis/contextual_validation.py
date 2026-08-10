"""Validation Context Module.

Integra análisis de alineaciones y cálculo de EV en un pipeline único.
"""

from typing import Dict, Optional, Tuple
from dataclasses import asdict

from engine.analysis.lineup_impact import LineupImpactModule, LineupImpact
from engine.analysis.fair_odds_engine import FairOddsEngine, EVAnalysisResult


class ContextualValidationPipeline:
    """Pipeline de validación contextual integrado."""
    
    def __init__(self):
        """Inicializar pipeline de validación."""
        self.lineup_module = LineupImpactModule()
        self.ev_engine = FairOddsEngine()
    
    def process_projections(
        self,
        projections: Dict[str, Dict],
        team_a_absences: Optional[list[Dict[str, str]]] = None,
        team_b_absences: Optional[list[Dict[str, str]]] = None,
        market_odds: Optional[Dict[str, float]] = None,
        bankroll: float = 100.0,
        team_a_context: str = "standard",
        team_b_context: str = "standard"
    ) -> EVAnalysisResult:
        """
        Procesa proyecciones a través del pipeline de validación completo.
        
        Args:
            projections: Proyecciones crudas del modelo
            team_a_absences: Ausencias equipo A
            team_b_absences: Ausencias equipo B
            market_odds: Cuotas reales del mercado
            bankroll: Bankroll para Kelly
            team_a_context: Contexto equipo A
            team_b_context: Contexto equipo B
            
        Returns:
            EVAnalysisResult: Resultado enriquecido con validación
        """
        # PASO 1: Evaluar impacto de alineaciones
        impact_a, impact_b = self.lineup_module.evaluate_lineup_impact(
            team_a_absences=team_a_absences,
            team_b_absences=team_b_absences,
            team_a_context=team_a_context,
            team_b_context=team_b_context
        )
        
        # PASO 2: Aplicar ajustes de alineación
        adjusted_projections = self.lineup_module.apply_lineup_impact_to_projections(
            projections,
            impact_a,
            impact_b
        )
        
        # PASO 3: Identificar picks con valor (EV+)
        value_picks = self.ev_engine.identify_value_picks(
            adjusted_projections,
            market_odds_dict=market_odds,
            bankroll=bankroll
        )
        
        # PASO 4: Generar resumen
        summary = self.ev_engine.generate_ev_report(
            projections,
            adjusted_projections,
            value_picks
        )
        
        # Agregar contexto de alineación al resumen
        summary['lineup_context'] = {
            'team_a': {
                'combined_factor': impact_a.combined_factor,
                'absent_count': len(impact_a.absent_key_players),
                'rationale': impact_a.rationale
            },
            'team_b': {
                'combined_factor': impact_b.combined_factor,
                'absent_count': len(impact_b.absent_key_players),
                'rationale': impact_b.rationale
            }
        }
        
        return EVAnalysisResult(
            raw_projections=projections,
            adjusted_projections=adjusted_projections,
            value_picks=value_picks,
            summary=summary
        )
    
    def to_dict(self, result: EVAnalysisResult) -> Dict:
        """
        Convierte EVAnalysisResult a diccionario estructurado.
        
        Args:
            result: Resultado del análisis
            
        Returns:
            Dict: Estructura JSON-compatible
        """
        return {
            'raw_projections': result.raw_projections,
            'adjusted_projections': result.adjusted_projections,
            'value_picks': [
                {
                    'market': pick.market,
                    'recommended_line': pick.recommended_line,
                    'fair_odds': pick.fair_odds,
                    'min_acceptable_odds': pick.min_acceptable_odds,
                    'ev_percentage': pick.ev_percentage,
                    'stake_percent': pick.stake_percent,
                    'rationale': pick.rationale,
                    'edge_magnitude': pick.edge_magnitude
                }
                for pick in result.value_picks
            ],
            'summary': result.summary
        }
