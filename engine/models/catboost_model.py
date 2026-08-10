import numpy as np
import math

class CatBoostPredictor:
    """CatBoost-based prediction model for football match statistics.
    
    Utiliza un enfoque de regresión escalable con fallback a promedios
    cuando no hay modelo entrenado disponible.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize CatBoost predictor.
        
        Args:
            model_path: Path to pre-trained model. If None, uses averaging mode.
        """
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = [
            'goals_a', 'goals_b', 'goals_max', 'goals_total',
            'corners_a', 'corners_b', 'corners_max', 'corners_total',
            'shots_on_target_a', 'shots_on_target_b', 'shots_on_target_max', 'shots_on_target_total',
            'total_shots_a', 'total_shots_b', 'total_shots_max', 'total_shots_total'
        ]
        self.markets = ['shots_on_target', 'total_shots', 'corners', 'goals']
        
        if model_path:
            try:
                self._load_catboost_model(model_path)
            except Exception as e:
                print(f"⚠️ No se pudo cargar modelo CatBoost: {e}")
                print("   Usando modo fallback (promedios ponderados)...")
                self.is_trained = False
    
    def _load_catboost_model(self, path):
        """Intentar cargar modelo CatBoost si está disponible."""
        try:
            from catboost import CatBoostRegressor
            import os
            
            if os.path.exists(path):
                self.model = CatBoostRegressor()
                self.model.load_model(path)
                self.model_path = path
                self.is_trained = True
                print(f"✅ Modelo CatBoost cargado desde: {path}")
        except ImportError:
            print("⚠️ CatBoost no está instalado. Usando fallback...")
        except Exception as e:
            print(f"⚠️ Error al cargar modelo: {e}")
    
    def _validate_features(self, features):
        """
        Validate that all required features are present.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            bool: True if all features are valid, False otherwise
        """
        required_keys = [f"{m}_{suffix}" for m in self.markets for suffix in ['a', 'b', 'max', 'total']]
        return all(key in features and features[key] is not None for key in required_keys)
    
    def _extract_feature_vector(self, features):
        """
        Extract feature vector in correct order for model.
        
        Args:
            features: Dictionary of features
            
        Returns:
            np.array: Feature vector
        """
        feature_vector = []
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            feature_vector.append(float(value) if value is not None else 0.0)
        return np.array([feature_vector])
    
    def _calculate_weighted_prediction(self, features, market):
        """
        Calcular predicción ponderada cuando no hay modelo entrenado.
        Utiliza características del mercado con pesos optimizados.
        
        Args:
            features: Feature dictionary
            market: Market name
            
        Returns:
            float: Predicted value
        """
        total_key = f"{market}_total"
        max_key = f"{market}_max"
        a_key = f"{market}_a"
        b_key = f"{market}_b"
        
        total = features.get(total_key, 0.0)
        max_val = features.get(max_key, total)
        val_a = features.get(a_key, 0.0)
        val_b = features.get(b_key, 0.0)
        
        # Ponderación: 60% promedio, 30% máximo, 10% asimetría
        base_prediction = total * 0.6 + (max_val * 2) * 0.25 + abs(val_a - val_b) * 0.15
        
        # Aplicar factor de momentum si existe
        momentum_key = f"{market}_momentum"
        momentum_factor = features.get(momentum_key, 1.0)
        if momentum_factor and momentum_factor != 0:
            base_prediction *= momentum_factor
        
        return max(0.0, base_prediction)
    
    def predict_market(self, features):
        """
        Predict market values using optimized algorithm.
        
        Args:
            features: Dictionary with combined team statistics
            
        Returns:
            Dictionary with predictions for each market
        """
        results = {}
        
        if not self._validate_features(features):
            print("⚠️ Advertencia: Características incompletas, usando valores por defecto.")
        
        for market in self.markets:
            total_key = f"{market}_total"
            max_key = f"{market}_max"
            
            if total_key not in features or features[total_key] is None:
                results[market] = self._fallback_prediction(market, features)
                continue
            
            try:
                # Si hay modelo entrenado, usarlo
                if self.is_trained and self.model:
                    feature_vector = self._extract_feature_vector(features)
                    prediction = self.model.predict(feature_vector)[0]
                    pred_total = max(0.0, prediction)
                else:
                    # Usar predicción ponderada sin modelo
                    pred_total = self._calculate_weighted_prediction(features, market)
                
                pred_max = features.get(max_key, features[total_key])
                
                # Aplicar líneas de apuesta profesionales
                safe_under = math.floor((pred_total - 1.5) * 2) / 2
                safe_over = math.ceil((pred_total + 1.5) * 2) / 2
                
                # Ajustar confianza según disponibilidad de modelo
                confidence = 0.85 if self.is_trained else 0.75
                
                results[market] = {
                    "projection_total": round(pred_total, 2),
                    "projection_max": round(pred_max, 2),
                    "safe_under_line": max(0.0, safe_under),
                    "safe_over_line": safe_over,
                    "model_confidence": confidence,
                    "model_type": "CatBoost" if self.is_trained else "Ponderado"
                }
            except Exception as e:
                print(f"⚠️ Error en predicción de {market}: {e}")
                results[market] = self._fallback_prediction(market, features)
        
        return results
    
    def _fallback_prediction(self, market, features):
        """
        Fallback prediction using simple averaging.
        
        Args:
            market: Market name
            features: Feature dictionary
            
        Returns:
            Dictionary with fallback prediction
        """
        total_key = f"{market}_total"
        max_key = f"{market}_max"
        
        pred_total = features.get(total_key, 0.0) or 0.0
        pred_max = features.get(max_key, pred_total) or pred_total
        
        safe_under = math.floor((pred_total - 1.5) * 2) / 2
        safe_over = math.ceil((pred_total + 1.5) * 2) / 2
        
        return {
            "projection_total": round(pred_total, 2),
            "projection_max": round(pred_max, 2),
            "safe_under_line": max(0.0, safe_under),
            "safe_over_line": safe_over,
            "model_confidence": 0.5,
            "model_type": "Fallback (Promedio)"
        }
    
    def train(self, X, y, eval_set=None):
        """
        Train the CatBoost model.
        
        Args:
            X: Training features (array-like)
            y: Training targets
            eval_set: Evaluation set for early stopping
        """
        try:
            from catboost import CatBoostRegressor
            
            self.model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                loss_function='RMSE',
                verbose=False,
                random_state=42,
                early_stopping_rounds=10
            )
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
            self.is_trained = True
            print("✅ Modelo CatBoost entrenado exitosamente")
        except ImportError:
            print("⚠️ CatBoost no disponible. Modo fallback activado.")
            self.is_trained = False
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model
        """
        if self.is_trained and self.model:
            try:
                self.model.save_model(path)
                self.model_path = path
                print(f"✅ Modelo guardado en: {path}")
            except Exception as e:
                print(f"⚠️ Error guardando modelo: {e}")
    
    def load_model(self, path):
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to model file
        """
        self._load_catboost_model(path)
