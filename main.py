import os
from engine.api_client import FootballDataClient
from engine.processor import DataProcessor
from engine.models.catboost_model import CatBoostPredictor
from engine.models.lstm_momentum import LSTMMomentumFilter

def run_pipeline(match_id):
    """
    Complete prediction pipeline using CatBoost + LSTM Momentum Filter.
    
    Args:
        match_id: ID of the match to predict
    """
    print(f"--- Iniciando Proyección Automática para Match ID: {match_id} ---")
    
    # Initialize components
    client = FootballDataClient()
    processor = DataProcessor()
    catboost_model = CatBoostPredictor()
    lstm_filter = LSTMMomentumFilter(sequence_length=5)

    # 1. Get match details
    print("\n[1/5] Obteniendo detalles del partido...")
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
    print("\n[2/5] Recopilando estadísticas históricas detalladas...")
    hist_a = client.get_historical_team_data(team_a_id, match_date)
    hist_b = client.get_historical_team_data(team_b_id, match_date)

    if not hist_a or not hist_b:
        print("⚠️ Advertencia: Datos históricos limitados")

    # 3. Calculate averages and prepare features
    print(f"\n[3/5] Procesando {len(hist_a)} partidos de {team_a_name} y {len(hist_b)} de {team_b_name}...")
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
    print("\n[4/5] Ejecutando modelo CatBoost...")
    catboost_predictions = catboost_model.predict_market(features)

    # Calculate momentum factors
    print("[4/5] Calculando momentum con LSTM...")
    momentum_a = lstm_filter.calculate_momentum_factor(hist_a, team_a_id)
    momentum_b = lstm_filter.calculate_momentum_factor(hist_b, team_b_id)
    
    # Average momentum factor
    combined_momentum = (momentum_a + momentum_b) / 2
    print(f"   📊 Momentum {team_a_name}: {momentum_a:.2f}")
    print(f"   📊 Momentum {team_b_name}: {momentum_b:.2f}")
    print(f"   📊 Momentum combinado: {combined_momentum:.2f}")

    # Apply LSTM momentum filter
    print("\n[5/5] Aplicando filtro de momentum...")
    final_predictions = lstm_filter.apply_momentum_filter(
        catboost_predictions, combined_momentum
    )

    # 5. Display results
    print(f"\n{'='*60}")
    print(f" 🎯 PROYECCIÓN FINAL: {team_a_name} vs {team_b_name}")
    print(f" 📅 Fecha: {match_date}")
    print(f" 🔄 Momentum: {combined_momentum:.2f} {final_predictions.get('goals', {}).get('momentum_direction', '')}")
    print(f"{'='*60}")
    
    for market, data in final_predictions.items():
        print(f"\n📈 Mercado: {market.upper()}")
        print(f"   ├─ Proyección Total: {data['projection_total']}")
        print(f"   ├─ Proyección Máximo: {data['projection_max']}")
        print(f"   ├─ LÍNEA BAJO (UNDER): < {data['safe_under_line']}")
        print(f"   ├─ LÍNEA ALTO (OVER): > {data['safe_over_line']}")
        print(f"   ├─ Confianza del modelo: {data['model_confidence']*100:.0f}%")
        print(f"   └─ Factor de momentum: {data['momentum_factor']}")
    
    print(f"\n{'='*60}")
    print("✅ Proyección completada exitosamente")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    M_ID = os.getenv('MATCH_ID')
    if not M_ID or M_ID == '0':
        print("❌ Error: Debes proporcionar un MATCH_ID.")
        print("   Uso: MATCH_ID=123456 python main.py")
    else:
        run_pipeline(M_ID)
