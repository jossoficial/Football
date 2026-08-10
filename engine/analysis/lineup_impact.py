"""Lineup Impact Analysis Module.

Evalúa el impacto de ausencias en atacantes clave y generadores de juego
sobre las proyecciones de partidos.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LineupImpact:
    """Estructura para almacenar datos de impacto de alineación."""
    attacking_impact: float  # Factor de reducción por ausencias en ataque
    creative_impact: float   # Factor de reducción por ausencias en creación
    combined_factor: float   # Factor combinado aplicable a proyecciones
    absent_key_players: list[str]  # Nombre de jugadores clave ausentes
    rationale: str  # Explicación del impacto


class LineupImpactModule:
    """Módulo de análisis de impacto de alineaciones."""
    
    # Configuración de jugadores clave por posición
    KEY_ATTACKERS = {'striker', 'forward', 'winger', 'right_winger', 'left_winger'}
    KEY_CREATORS = {'playmaker', 'midfielder', 'creative_midfielder', '10', 'center_midfielder'}
    
    # Factores de reducción por ausencia
    ATTACKING_IMPACT_FACTOR: float = -0.12  # -12% por falta de delantero
    CREATIVE_IMPACT_FACTOR: float = -0.10   # -10% por falta de creatividad
    
    # Umbral mínimo de factor aplicable (no reducir más del 25%)
    MIN_COMBINED_FACTOR: float = 0.75
    
    def __init__(self):
        """Inicializar módulo de impacto de alineaciones."""
        pass
    
    def evaluate_lineup_impact(
        self,
        team_a_absences: Optional[list[Dict[str, str]]] = None,
        team_b_absences: Optional[list[Dict[str, str]]] = None,
        team_a_context: Optional[str] = None,
        team_b_context: Optional[str] = None
    ) -> Tuple[LineupImpact, LineupImpact]:
        """
        Evalúa el impacto de ausencias en alineaciones.
        
        Args:
            team_a_absences: Lista de dict con {'player': str, 'position': str}
            team_b_absences: Lista de dict con {'player': str, 'position': str}
            team_a_context: Contexto adicional del equipo A
            team_b_context: Contexto adicional del equipo B
            
        Returns:
            Tuple[LineupImpact, LineupImpact]: Impacto para ambos equipos
        """
        impact_a = self._calculate_team_impact(
            team_a_absences or [],
            team_a_context or "standard"
        )
        impact_b = self._calculate_team_impact(
            team_b_absences or [],
            team_b_context or "standard"
        )
        
        return impact_a, impact_b
    
    def _calculate_team_impact(
        self,
        absences: list[Dict[str, str]],
        context: str
    ) -> LineupImpact:
        """
        Calcula el impacto para un equipo específico.
        
        Args:
            absences: Lista de jugadores ausentes con posición
            context: Contexto del equipo (standard, injury_crisis, etc)
            
        Returns:
            LineupImpact: Estructura con factores de impacto
        """
        attacking_reduction = 0.0
        creative_reduction = 0.0
        absent_players = []
        
        for absence in absences:
            player_name = absence.get('player', 'unknown')
            position = absence.get('position', '').lower()
            
            # Evaluar si es delantero clave
            if any(pos in position for pos in self.KEY_ATTACKERS):
                attacking_reduction += abs(self.ATTACKING_IMPACT_FACTOR)
                absent_players.append(f"{player_name} ({position})")
            
            # Evaluar si es creador clave
            if any(pos in position for pos in self.KEY_CREATORS):
                creative_reduction += abs(self.CREATIVE_IMPACT_FACTOR)
                if f"{player_name} ({position})" not in absent_players:
                    absent_players.append(f"{player_name} ({position})")
        
        # Aplicar contexto adicional
        context_multiplier = self._get_context_multiplier(context)
        attacking_reduction *= context_multiplier
        creative_reduction *= context_multiplier
        
        # Calcular factor combinado (NO es suma, es multiplicativo)
        combined_factor = (1.0 - attacking_reduction) * (1.0 - creative_reduction)
        combined_factor = max(combined_factor, self.MIN_COMBINED_FACTOR)
        
        # Generar rationale
        rationale = self._generate_rationale(
            attacking_reduction,
            creative_reduction,
            absent_players,
            context
        )
        
        return LineupImpact(
            attacking_impact=1.0 - attacking_reduction,
            creative_impact=1.0 - creative_reduction,
            combined_factor=combined_factor,
            absent_key_players=absent_players,
            rationale=rationale
        )
    
    def _get_context_multiplier(self, context: str) -> float:
        """
        Obtiene multiplicador de contexto.
        
        Args:
            context: Contexto del equipo
            
        Returns:
            float: Multiplicador aplicable
        """
        context_map = {
            'standard': 1.0,
            'injury_crisis': 1.5,
            'reserves': 1.3,
            'rotation': 0.8,
            'key_player_out': 1.2
        }
        return context_map.get(context.lower(), 1.0)
    
    def _generate_rationale(self, attacking_red: float, creative_red: float,
                            absent: list[str], context: str) -> str:
        """
        Genera explicación textual del impacto.
        
        Args:
            attacking_red: Reducción en ataque (0.0-1.0)
            creative_red: Reducción en creatividad (0.0-1.0)
            absent: Lista de jugadores ausentes
            context: Contexto
            
        Returns:
            str: Explicación del impacto
        """
        components = []
        
        if attacking_red > 0.0:
            components.append(f"Ataque reducido {attacking_red*100:.0f}%")
        if creative_red > 0.0:
            components.append(f"Creatividad reducida {creative_red*100:.0f}%")
        
        if absent:
            components.append(f"Ausentes: {len(absent)} clave")
        
        if context != 'standard':
            components.append(f"Contexto: {context}")
        
        return " | ".join(components) if components else "Sin impacto material"
    
    def apply_lineup_impact_to_projections(
        self,
        projections: Dict[str, Dict],
        impact_a: LineupImpact,
        impact_b: LineupImpact
    ) -> Dict[str, Dict]:
        """
        Aplica factores de impacto a las proyecciones del modelo.
        
        Args:
            projections: Proyecciones crudas del modelo
            impact_a: Impacto equipo A
            impact_b: Impacto equipo B
            
        Returns:
            Dict: Proyecciones ajustadas
        """
        adjusted = {}
        
        # Usar promedio de ambos equipos para factor defensivo/ofensivo
        combined_impact = (impact_a.combined_factor + impact_b.combined_factor) / 2.0
        
        for market, pred_data in projections.items():
            adjusted_proj = pred_data.copy()
            
            # Aplicar factor solo a métricas ofensivas
            if market in ['shots_on_target', 'total_shots', 'corners', 'goals']:
                adjustment_factor = combined_impact
                
                adjusted_proj['projection_total'] = round(
                    pred_data['projection_total'] * adjustment_factor, 2
                )
                adjusted_proj['projection_max'] = round(
                    pred_data['projection_max'] * adjustment_factor, 2
                )
                
                # Recalcular líneas de apuesta
                import math
                total = adjusted_proj['projection_total']
                safe_under = math.floor((total - 1.5) * 2) / 2
                safe_over = math.ceil((total + 1.5) * 2) / 2
                
                adjusted_proj['safe_under_line'] = max(0.0, safe_under)
                adjusted_proj['safe_over_line'] = safe_over
                adjusted_proj['lineup_adjustment_factor'] = round(adjustment_factor, 3)
            
            adjusted[market] = adjusted_proj
        
        return adjusted
