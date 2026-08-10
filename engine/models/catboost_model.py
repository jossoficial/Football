import numpy as np
from catboost import CatBoostRegressor
import joblib
import os
from pathlib import Path

class CatBoostPredictor:
    """CatBoost-based prediction model for football match statistics."""
    
    def __init__(self, model_path=None):
        """
        Initialize CatBoost predictor.
        
        Args:
            model_path: Path to pre-trained model. If None, creates new model.
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'goals_a', 'goals_b', 'goals_max', 'goals_total',
            'corners_a', 'corners_b', 'corners_max', 'corners_total',
            'shots_on_target_a', 'shots_on_target_b', 'shots_on_target_max', 'shots_on_target_total',
            'total_shots_a', 'total_shots_b', 'total_shots_max', 'total_shots_total'
        ]
        self.markets = ['shots_on_target', 'total_shots', 'corners', 'goals']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a new CatBoost regressor."""
        self.model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            verbose=False,
            random_state=42,
            early_stopping_rounds=10
        )
    
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
    
    def predict_market(self, features):
        """
        Predict market values using CatBoost model.
        
        Args:
            features: Dictionary with combined team statistics
            
        Returns:
            Dictionary with predictions for each market
        """
        results = {}
        
        if not self._validate_features(features):
            print("⚠️ Advertencia: Características incompletas, usando valores por defecto.")
        
        feature_vector = self._extract_feature_vector(features)
        
        for market in self.markets:
            # Si no hay suficientes datos, usar fallback a promedio
            total_key = f"{market}_total"
            max_key = f"{market}_max"
            
            if total_key not in features or features[total_key] is None:
                results[market] = self._fallback_prediction(market, features)
                continue
            
            try:
                # Predicción del modelo CatBoost
                if self.model and hasattr(self.model, 'predict'):
                    prediction = self.model.predict(feature_vector)[0]
                    pred_total = max(0.0, prediction)  # Asegurar que no sea negativo
                else:
                    # Si no hay modelo entrenado, usar promedio
                    pred_total = features[total_key]
                
                pred_max = features.get(max_key, features[total_key])
                
                # Aplicar líneas de apuesta profesionales
                import math
                safe_under = math.floor((pred_total - 1.5) * 2) / 2
                safe_over = math.ceil((pred_total + 1.5) * 2) / 2
                
                results[market] = {
                    "projection_total": round(pred_total, 2),
                    "projection_max": round(pred_max, 2),
                    "safe_under_line": max(0.0, safe_under),
                    "safe_over_line": safe_over,
                    "model_confidence": 0.8  # Confianza del modelo
                }
            except Exception as e:
                print(f"❌ Error en predicción de {market}: {e}")
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
        import math
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
            "model_confidence": 0.5  # Confianza baja (fallback)
        }
    
    def train(self, X, y, eval_set=None):
        """
        Train the CatBoost model.
        
        Args:
            X: Training features (array-like)
            y: Training targets
            eval_set: Evaluation set for early stopping
        """
        if self.model:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model:
            self.model.save_model(path)
            self.model_path = path
    
    def load_model(self, path):
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to model file
        """
        if os.path.exists(path):
            self.model = CatBoostRegressor()
            self.model.load_model(path)
            self.model_path = path
