import os
import sys
import json
from engine.api_client import FootballDataClient
from engine.processor import DataProcessor
from engine.models.catboost_model import CatBoostPredictor
from engine.models.lstm_momentum import LSTMMomentumFilter
from engine.analysis.contextual_validation import ContextualValidationPipeline

def run_pipeline(match_id):
    """
    Complete prediction pipeline using CatBoost + LSTM + Contextual Validation.
    
    Args:
        match_id: ID of the match to predict
    """
    print(f"--- Iniciando Proyección Automática para Match ID: {match_id} ---")
    
    # Initialize components
    client = FootballDataClient()
    processor = DataProcessor()
    catboost_model = CatBoostPredictor()
    lstm_filter = LSTMMomentumFilter(sequence_length=5)
    validation_pipeline = ContextualValidationPipeline()

    # 1. Get match details
    print("\n[1/6] Obteniendo detalles del partido...")
    response = client.get_match_details(match_id)
    if not response:
        print("❌ Error: No se pudo obtener la respuesta de la API.")
        return

    match_data = response.get('data', response)

    # Extract team IDs and names
    team_a_id = match_data.get('home_team', {}).get('id')
    team_b_id = match_data.get('away_team', {}).get('id')
    team_a_name = match_data.get('home_team', {}).get('name', 'Local')
    team_b_name = match_data.get('away_team', {}).get('name', 'Visitante')
    match_date = match_data.get('utc_date', '').split('T')[0]

    if not team_a_id or not team_b_id:
        print("❌ Error: No se pudieron identificar los IDs de los equipos.")
        return

    print(f"✅ Partido: {team_a_name} vs {team_b_name} ({match_date})")

    # 2. Collect historical data
    print("\n[2/6] Recopilando estadísticas históricas detalladas...")
    hist_a = client.get_historical_team_data(team_a_id, match_date)
    hist_b = client.get_historical_team_data(team_b_id, match_date)

    if not hist_a or not hist_b:
        print("⚠️ Advertencia: Datos históricos limitados")

    # 3. Calculate averages and prepare features
    print(f"\n[3/6] Procesando {len(hist_a)} partidos de {team_a_name} y {len(hist_b)} de {team_b_name}...")
    avg_a = processor.calculate_averages(hist_a, team_a_id)
    avg_b = processor.calculate_averages(hist_b, team_b_id)

    if not avg_a or not avg_b:
        print("❌ Error: No se pudieron calcular promedios suficientes.")
        return

    # Prepare base features
    features = processor.prepare_features(avg_a, avg_b)
    
    # Enrich features with momentum indicators
    features = processor.enrich_features_with_momentum(
        features, hist_a, hist_b, team_a_id, team_b_id
    )

    # 4. CatBoost Prediction
    print("\n[4/6] Ejecutando modelo CatBoost...")
    catboost_predictions = catboost_model.predict_market(features)

    # Calculate momentum factors
    print("[4/6] Calculando momentum con LSTM...")
    momentum_a = lstm_filter.calculate_momentum_factor(hist_a, team_a_id)
    momentum_b = lstm_filter.calculate_momentum_factor(hist_b, team_b_id)
    
    # Average momentum factor
    combined_momentum = (momentum_a + momentum_b) / 2
    print(f"   📊 Momentum {team_a_name}: {momentum_a:.2f}")
    print(f"   📊 Momentum {team_b_name}: {momentum_b:.2f}")
    print(f"   📊 Momentum combinado: {combined_momentum:.2f}")

    # Apply LSTM momentum filter
    print("\n[5/6] Aplicando filtro de momentum...")
    final_predictions = lstm_filter.apply_momentum_filter(
        catboost_predictions, combined_momentum
    )

    # 5. Contextual Validation & EV Analysis
    print("\n[5/6] Aplicando validación contextual y análisis de valor...")
    
    # Datos de alineación (ejemplo - en producción, obtendría del API/BD)
    team_a_absences = []  # {'player': 'Nombre', 'position': 'Striker'}
    team_b_absences = []  # {'player': 'Nombre', 'position': 'Midfielder'}
    
    # Cuotas del mercado (ejemplo - en producción, del API de apuestas)
    market_odds = {
        'goals': 2.45,
        'shots_on_target': 1.95,
        'corners': 3.20,
        'total_shots': 2.10
    }
    
    ev_result = validation_pipeline.process_projections(
        projections=final_predictions,
        team_a_absences=team_a_absences,
        team_b_absences=team_b_absences,
        market_odds=market_odds,
        bankroll=100.0,
        team_a_context="standard",
        team_b_context="standard"
    )
    
    # 6. Display results
    print(f"\n[6/6] Generando reporte final...")
    print(f"\n{'='*70}")
    print(f" 🎯 PROYECCIÓN FINAL ENRIQUECIDA: {team_a_name} vs {team_b_name}")
    print(f" 📅 Fecha: {match_date}")
    print(f" 🔄 Momentum: {combined_momentum:.2f} {final_predictions.get('goals', {}).get('momentum_direction', '')}")
    print(f"{'='*70}")
    
    # Raw projections
    print("\n📑 PROYECCIONES CRUDAS:")
    for market, data in ev_result.raw_projections.items():
        print(f"  {market.upper()}: {data.get('projection_total', 0)} | Confianza: {data.get('model_confidence', 0)*100:.0f}%")
    
    # Adjusted projections
    print("\n📊 PROYECCIONES AJUSTADAS (post alineación):")
    for market, data in ev_result.adjusted_projections.items():
        if 'lineup_adjustment_factor' in data:
            print(f"  {market.upper()}: {data.get('projection_total', 0)} | Ajuste: {data['lineup_adjustment_factor']:.3f}")
        else:
            print(f"  {market.upper()}: {data.get('projection_total', 0)}")
    
    # Value picks
    print(f"\n🚀 PICKS CON VALOR (EV+): {len(ev_result.value_picks)} identificados")
    for i, pick in enumerate(ev_result.value_picks, 1):
        print(f"\n  [{i}] {pick.market.upper()}")
        print(f"      Línea Recomendada: {pick.recommended_line}")
        print(f"      Cuota Justa: {pick.fair_odds:.2f}")
        print(f"      Mínima Aceptable: {pick.min_acceptable_odds:.2f}")
        print(f"      EV: {pick.ev_percentage:+.2f}%")
        print(f"      Stake Sugerido: {pick.stake_percent:.2f}% del bankroll")
        print(f"      Razonamiento: {pick.rationale}")
    
    # Summary
    print(f"\n📈 RESUMEN EJECUTIVO:")
    summary = ev_result.summary
    print(f"  Total de picks con valor: {summary.get('total_picks_identified', 0)}")
    print(f"  EV total acumulado: {summary.get('total_positive_ev_percentage', 0):+.2f}%")
    print(f"  Stake total recomendado: {summary.get('total_recommended_stake', 0):.2f}%")
    print(f"  Mejor pick EV: {summary.get('best_pick_ev', 0):+.2f}%")
    print(f"  Mercados analizados: {summary.get('markets_analyzed', 0)}")
    
    # Lineup context
    lineup_ctx = summary.get('lineup_context', {})
    if lineup_ctx:
        print(f"\n👥 CONTEXTO DE ALINEACIÓN:")
        print(f"  {team_a_name}: Factor {lineup_ctx.get('team_a', {}).get('combined_factor', 1.0):.3f} | {lineup_ctx.get('team_a', {}).get('rationale', 'Sin impacto')}")
        print(f"  {team_b_name}: Factor {lineup_ctx.get('team_b', {}).get('combined_factor', 1.0):.3f} | {lineup_ctx.get('team_b', {}).get('rationale', 'Sin impacto')}")
    
    print(f"\n{'='*70}")
    print("✅ Proyección completada exitosamente")
    print(f"{'='*70}\n")
    
    # Salida JSON
    output_dict = validation_pipeline.to_dict(ev_result)
    print("\n💾 SALIDA JSON ESTRUCTURADA:")
    print(json.dumps(output_dict, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    M_ID = os.getenv('MATCH_ID')
    
    # 1. MODO MANUAL INDIVIDUAL
    if M_ID and M_ID != '0' and M_ID.strip() != "":
        print(f"Modo manual activado por variable de entorno MATCH_ID.")
        run_pipeline(M_ID)
        
    # 2. MODO AUTOMÁTICO (ESCANEO DE JORNADA COMPLETA)
    else:
        print("🤖 MATCH_ID no detectado o vacío. Iniciando Escaneo Automático de la jornada...")
        
        try:
            client = FootballDataClient()
            
            # Consulta los partidos agendados para la fecha de hoy
            response_jornada = client.get_today_matches()
            
            if not response_jornada:
                print("⚠️ No se encontraron partidos programados o la API no devolvió datos para el día de hoy.")
                sys.exit(0)
                
            # Extrae la lista de partidos del diccionario de respuesta
            partidos = response_jornada.get('matches', response_jornada) if isinstance(response_jornada, dict) else response_jornada
            
            if not isinstance(partidos, list) or len(partidos) == 0:
                print("⚠️ La respuesta de la jornada no contiene una lista de partidos procesable.")
                sys.exit(0)

            print(f"📊 Se identificaron {len(partidos)} partidos para procesar hoy en los modelos predictivos.")
            
            # Bucle de análisis automatizado por lote
            for idx, partido in enumerate(partidos, 1):
                partido_id = partido.get('id')
                if not partido_id:
                    continue
                    
                print(f"\n🚀 [{idx}/{len(partidos)}] Procesando partido en segundo plano...")
                try:
                    run_pipeline(str(partido_id))
                except Exception as e:
                    print(f"❌ Error procesando el partido ID {partido_id}: {str(e)}")
                    print("⏩ Saltando al siguiente evento de la lista...")
                    continue
                    
            print("\n✅ Escaneo automático de la jornada finalizado con éxito.")
            
        except AttributeError:
            print("❌ Error: 'FootballDataClient' requiere un método 'get_today_matches()' válido para listar los eventos del día.")
            print("   Por favor, revisa tus funciones en 'engine/api_client.py'.")
        except Exception as e:
            print(f"❌ Error crítico en el escaneo por lote: {str(e)}")
