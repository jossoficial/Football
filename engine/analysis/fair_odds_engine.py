"""Fair Odds Engine y Expected Value Module.

Cálculo de cuotas justas, EV+ y stake sugerido usando criterio de Kelly fraccionado.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ValuePick:
    """Estructura para una selección con valor."""
    market: str
    recommended_line: float
    fair_odds: float
    min_acceptable_odds: float
    ev_percentage: float
    stake_percent: float
    rationale: str
    edge_magnitude: float  # Diferencia en decimales vs fair_odds


@dataclass
class EVAnalysisResult:
    """Estructura para resultado completo de análisis EV."""
    raw_projections: Dict
    adjusted_projections: Dict
    value_picks: list[ValuePick]
    summary: Dict


class FairOddsEngine:
    """Motor de cálculo de cuotas justas y EV+."""
    
    # Configuración de thresholds
    MIN_EV_THRESHOLD: float = 0.02  # 2% mínimo de EV para considerar
    MIN_ACCEPTABLE_EV: float = 0.01  # 1% mínimo aceptable en mercado
    KELLY_FRACTION: float = 0.25     # Kelly fraccionado al 25%
    MAX_STAKE_PERCENT: float = 0.03  # Máximo 3% del bankroll
    
    # Confidence to probability mapping
    CONFIDENCE_TO_PROB_MAP = {
        0.50: 0.45,  # Confidence baja -> probabilidad conservadora
        0.60: 0.52,
        0.70: 0.62,
        0.75: 0.68,
        0.80: 0.75,
        0.85: 0.82,
        0.90: 0.88,
        0.95: 0.94
    }
    
    def __init__(self):
        """Inicializar motor de cuotas justas."""
        pass
    
    def confidence_to_probability(
        self,
        confidence: float
    ) -> float:
        """
        Convierte confianza del modelo a probabilidad.
        
        Args:
            confidence: Confianza 0.0-1.0
            
        Returns:
            float: Probabilidad estimada 0.0-1.0
        """
        if confidence <= 0.0 or confidence > 1.0:
            raise ValueError(f"Confidence debe estar en (0.0, 1.0], recibido: {confidence}")
        
        # Interpolación lineal entre puntos conocidos
        sorted_keys = sorted(self.CONFIDENCE_TO_PROB_MAP.keys())
        
        if confidence <= sorted_keys[0]:
            return self.CONFIDENCE_TO_PROB_MAP[sorted_keys[0]]
        if confidence >= sorted_keys[-1]:
            return self.CONFIDENCE_TO_PROB_MAP[sorted_keys[-1]]
        
        # Encontrar intervalo e interpolar
        for i in range(len(sorted_keys) - 1):
            if sorted_keys[i] <= confidence <= sorted_keys[i + 1]:
                c1, c2 = sorted_keys[i], sorted_keys[i + 1]
                p1, p2 = self.CONFIDENCE_TO_PROB_MAP[c1], self.CONFIDENCE_TO_PROB_MAP[c2]
                
                # Interpolación lineal
                prob = p1 + (confidence - c1) * (p2 - p1) / (c2 - c1)
                return min(prob, 0.99)  # Cap en 0.99
        
        return self.CONFIDENCE_TO_PROB_MAP[sorted_keys[-1]]
    
    def calculate_fair_odds(
        self,
        probability: float
    ) -> float:
        """
        Calcula la cuota justa (fair odds) a partir de probabilidad.
        
        Args:
            probability: Probabilidad del evento 0.0-1.0
            
        Returns:
            float: Cuota justa en formato decimal
            
        Raises:
            ValueError: Si probabilidad es inválida
        """
        if probability <= 0.0 or probability >= 1.0:
            raise ValueError(f"Probabilidad debe estar en (0.0, 1.0), recibido: {probability}")
        
        fair_odds = 1.0 / probability
        return round(fair_odds, 2)
    
    def calculate_ev(
        self,
        probability: float,
        market_odds: float
    ) -> float:
        """
        Calcula el Valor Esperado (EV).
        
        Args:
            probability: Probabilidad del evento
            market_odds: Cuota del mercado (decimal)
            
        Returns:
            float: EV decimal (ej: 0.05 = +5%)
            
        Raises:
            ValueError: Si inputs son inválidos
        """
        if probability <= 0.0 or probability >= 1.0:
            raise ValueError(f"Probabilidad inválida: {probability}")
        if market_odds <= 1.0:
            raise ValueError(f"Cuota debe ser > 1.0, recibido: {market_odds}")
        
        ev = (probability * market_odds) - 1.0
        return round(ev, 4)
    
    def calculate_kelly_stake(
        self,
        probability: float,
        market_odds: float,
        bankroll: float = 100.0
    ) -> Tuple[float, float]:
        """
        Calcula el stake usando criterio de Kelly fraccionado.
        
        Args:
            probability: Probabilidad del evento
            market_odds: Cuota del mercado
            bankroll: Bankroll disponible (default 100)
            
        Returns:
            Tuple[float, float]: (stake_absolute, stake_percentage)
            
        Raises:
            ValueError: Si inputs son inválidos
        """
        if probability <= 0.0 or probability >= 1.0:
            raise ValueError(f"Probabilidad inválida: {probability}")
        if market_odds <= 1.0:
            raise ValueError(f"Cuota inválida: {market_odds}")
        if bankroll <= 0.0:
            raise ValueError(f"Bankroll debe ser positivo: {bankroll}")
        
        # Criterio de Kelly: f* = (bp - q) / b, donde b=odds-1, q=1-p
        b = market_odds - 1.0
        q = 1.0 - probability
        
        # Kelly puro
        kelly_fraction = (b * probability - q) / b
        
        # Validar que EV sea positivo
        if kelly_fraction <= 0.0:
            return 0.0, 0.0
        
        # Aplicar fracción conservadora (25% de Kelly)
        fractional_kelly = kelly_fraction * self.KELLY_FRACTION
        
        # Aplicar límite máximo
        stake_percentage = min(fractional_kelly, self.MAX_STAKE_PERCENT)
        stake_absolute = bankroll * stake_percentage
        
        return round(stake_absolute, 2), round(stake_percentage, 4)
    
    def identify_value_picks(
        self,
        projections: Dict[str, Dict],
        market_odds_dict: Optional[Dict[str, float]] = None,
        bankroll: float = 100.0
    ) -> list[ValuePick]:
        """
        Identifica selecciones con valor (EV+).
        
        Args:
            projections: Proyecciones del modelo
            market_odds_dict: Dict con cuotas reales {'market': odds}
            bankroll: Bankroll para cálculo de Kelly
            
        Returns:
            list[ValuePick]: Selecciones ordenadas por EV
        """
        value_picks = []
        
        for market, pred_data in projections.items():
            confidence = pred_data.get('model_confidence', 0.75)
            
            try:
                # Convertir confianza a probabilidad
                probability = self.confidence_to_probability(confidence)
                
                # Calcular cuota justa
                fair_odds = self.calculate_fair_odds(probability)
                
                # Obtener cuota del mercado (si no existe, usar fair_odds)
                market_odds = market_odds_dict.get(market, fair_odds) if market_odds_dict else fair_odds
                
                # Calcular EV
                ev = self.calculate_ev(probability, market_odds)
                
                # Solo incluir si EV > threshold
                if ev < self.MIN_EV_THRESHOLD:
                    continue
                
                # Calcular stake sugerido
                stake_absolute, stake_percent = self.calculate_kelly_stake(
                    probability, market_odds, bankroll
                )
                
                # Línea recomendada es la proyección total
                recommended_line = pred_data.get('projection_total', 0.0)
                
                # Calcular mínimo de cuota aceptable (EV = 1%)
                min_acceptable_odds = self._calculate_min_acceptable_odds(
                    probability,
                    self.MIN_ACCEPTABLE_EV
                )
                
                # Edge magnitude
                edge = market_odds - fair_odds
                
                # Generar rationale
                rationale = self._generate_value_rationale(
                    market, probability, market_odds, fair_odds, ev
                )
                
                pick = ValuePick(
                    market=market,
                    recommended_line=round(recommended_line, 2),
                    fair_odds=fair_odds,
                    min_acceptable_odds=round(min_acceptable_odds, 2),
                    ev_percentage=round(ev * 100, 2),
                    stake_percent=stake_percent * 100,
                    rationale=rationale,
                    edge_magnitude=round(edge, 2)
                )
                
                value_picks.append(pick)
            
            except (ValueError, ZeroDivisionError) as e:
                # Saltar picks con datos inválidos
                continue
        
        # Ordenar por EV descendente
        value_picks.sort(key=lambda x: x.ev_percentage, reverse=True)
        
        return value_picks
    
    def _calculate_min_acceptable_odds(
        self,
        probability: float,
        ev_target: float
    ) -> float:
        """
        Calcula la mínima cuota para alcanzar un EV objetivo.
        
        Args:
            probability: Probabilidad del evento
            ev_target: EV objetivo (ej: 0.01 = 1%)
            
        Returns:
            float: Mínima cuota requerida
        """
        # EV = (p * odds) - 1, despejamos odds = (EV + 1) / p
        min_odds = (ev_target + 1.0) / probability
        return min_odds
    
    def _generate_value_rationale(
        self,
        market: str,
        probability: float,
        market_odds: float,
        fair_odds: float,
        ev: float
    ) -> str:
        """
        Genera rationale cuantitativa para el pick.
        
        Args:
            market: Nombre del mercado
            probability: Probabilidad
            market_odds: Cuota del mercado
            fair_odds: Cuota justa
            ev: Valor esperado
            
        Returns:
            str: Rationale cuantitativa
        """
        edge = market_odds - fair_odds
        return f"Prob {probability*100:.1f}% | Fair {fair_odds:.2f} | Mkt {market_odds:.2f} | Edge +{edge:.2f} | EV {ev*100:+.1f}%"
    
    def generate_ev_report(
        self,
        raw_projections: Dict,
        adjusted_projections: Dict,
        value_picks: list[ValuePick]
    ) -> Dict:
        """
        Genera reporte resumido de EV.
        
        Args:
            raw_projections: Proyecciones crudas
            adjusted_projections: Proyecciones ajustadas
            value_picks: Selecciones con valor
            
        Returns:
            Dict: Resumen ejecutivo
        """
        total_ev = sum(pick.ev_percentage for pick in value_picks)
        total_stake = sum(pick.stake_percent for pick in value_picks)
        
        return {
            'total_picks_identified': len(value_picks),
            'total_positive_ev_percentage': round(total_ev, 2),
            'total_recommended_stake': round(total_stake, 2),
            'best_pick_ev': round(value_picks[0].ev_percentage, 2) if value_picks else 0.0,
            'markets_analyzed': len(adjusted_projections),
            'markets_with_value': len(value_picks)
        }
