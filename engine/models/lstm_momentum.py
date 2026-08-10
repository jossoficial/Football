import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
from pathlib import Path

class LSTMMomentumFilter:
    """LSTM-based momentum filter for football match predictions.
    
    Analyzes temporal patterns in team performance to adjust predictions
    based on recent form and momentum.
    """
    
    def __init__(self, sequence_length=5, model_path=None):
        """
        Initialize LSTM momentum filter.
        
        Args:
            sequence_length: Number of previous matches to consider (default: 5)
            model_path: Path to pre-trained LSTM model
        """
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler_dict = {}  # Store scalers for normalization
        self.markets = ['shots_on_target', 'total_shots', 'corners', 'goals']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_lstm_model()
    
    def _build_lstm_model(self):
        """Build LSTM neural network architecture."""
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, 4), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Output: momentum factor (0.8 to 1.2)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def _extract_momentum_sequence(self, history, team_id):
        """
        Extract sequence of performance metrics from match history.
        
        Args:
            history: List of match data dictionaries
            team_id: Team ID to extract stats for
            
        Returns:
            np.array: Sequence of shape (sequence_length, 4) or None if insufficient data
        """
        if not history or len(history) < self.sequence_length:
            return None
        
        sequence = []
        metrics = ['goals', 'corners', 'shots_on_target', 'total_shots']
        
        # Extract last sequence_length matches
        for item in history[-self.sequence_length:]:
            if not item:
                continue
            
            match = item.get('match', {})
            stats = item.get('stats', {})
            
            # Determine if team is home or away
            home_team = match.get('home_team', {}) or {}
            away_team = match.get('away_team', {}) or {}
            home_id = str(home_team.get('id', ''))
            away_id = str(away_team.get('away_id', ''))
            side = 'home' if str(team_id) == home_id else 'away'
            
            # Extract metric values
            overview = stats.get('overview', {}) or {}
            score = match.get('score', {}) or {}
            
            row = []
            for metric in metrics:
                if metric == 'goals':
                    value = score.get(side)
                else:
                    stat_obj = overview.get(self._metric_to_api_name(metric), {}) or {}
                    all_obj = stat_obj.get('all', {}) or {}
                    value = all_obj.get(side)
                
                row.append(float(value) if isinstance(value, (int, float)) else 0.0)
            
            sequence.append(row)
        
        if len(sequence) < self.sequence_length:
            return None
        
        return np.array(sequence[-self.sequence_length:])
    
    def _metric_to_api_name(self, metric):
        """Convert metric name to API field name."""
        mapping = {
            'goals': 'goals',
            'corners': 'corner_kicks',
            'shots_on_target': 'shots_on_target',
            'total_shots': 'total_shots'
        }
        return mapping.get(metric, metric)
    
    def _normalize_sequence(self, sequence):
        """
        Normalize sequence values to [0, 1] range.
        
        Args:
            sequence: np.array of shape (sequence_length, 4)
            
        Returns:
            np.array: Normalized sequence
        """
        if sequence is None:
            return None
        
        normalized = np.zeros_like(sequence, dtype=float)
        for i in range(sequence.shape[1]):
            col_max = np.max(sequence[:, i])
            col_min = np.min(sequence[:, i])
            
            if col_max > col_min:
                normalized[:, i] = (sequence[:, i] - col_min) / (col_max - col_min)
            else:
                normalized[:, i] = sequence[:, i]
        
        return normalized
    
    def calculate_momentum_factor(self, history, team_id):
        """
        Calculate momentum factor based on recent performance trends.
        
        Args:
            history: List of match data
            team_id: Team ID
            
        Returns:
            float: Momentum factor (0.8 to 1.2), where 1.0 = neutral
        """
        sequence = self._extract_momentum_sequence(history, team_id)
        
        if sequence is None:
            # Default momentum if insufficient history
            return 1.0
        
        try:
            # Normalize sequence
            normalized = self._normalize_sequence(sequence)
            
            # Predict momentum factor using LSTM
            if self.model:
                momentum = self.model.predict(np.array([normalized]), verbose=0)[0][0]
                # Clamp momentum factor to reasonable range
                momentum_factor = np.clip(momentum, 0.8, 1.2)
            else:
                # If no model, calculate momentum from trend
                momentum_factor = self._calculate_trend_momentum(sequence)
            
            return float(momentum_factor)
        
        except Exception as e:
            print(f"⚠️ Error calculando momentum: {e}")
            return 1.0
    
    def _calculate_trend_momentum(self, sequence):
        """
        Calculate momentum using simple trend analysis.
        
        Args:
            sequence: Performance sequence
            
        Returns:
            float: Momentum factor
        """
        # Average of last 2 matches vs average of first 3 matches
        recent_avg = np.mean(sequence[-2:, :].flatten())
        historical_avg = np.mean(sequence[:3, :].flatten())
        
        if historical_avg > 0:
            trend_ratio = recent_avg / historical_avg
            return np.clip(trend_ratio, 0.8, 1.2)
        return 1.0
    
    def apply_momentum_filter(self, predictions, momentum_factor):
        """
        Apply momentum factor to adjust predictions.
        
        Args:
            predictions: Dictionary of market predictions from CatBoost
            momentum_factor: Momentum multiplier (0.8 to 1.2)
            
        Returns:
            Dictionary: Adjusted predictions
        """
        import math
        filtered_predictions = {}
        
        for market, pred_data in predictions.items():
            # Apply momentum factor to projections
            adjusted_total = pred_data['projection_total'] * momentum_factor
            adjusted_max = pred_data['projection_max'] * momentum_factor
            
            # Recalculate safe lines with adjusted values
            safe_under = math.floor((adjusted_total - 1.5) * 2) / 2
            safe_over = math.ceil((adjusted_total + 1.5) * 2) / 2
            
            # Calculate confidence boost from momentum
            base_confidence = pred_data.get('model_confidence', 0.8)
            momentum_confidence = 0.9 + (abs(momentum_factor - 1.0) * 0.1)  # Confidence increases with strong momentum
            combined_confidence = min(0.95, (base_confidence + momentum_confidence) / 2)
            
            filtered_predictions[market] = {
                "projection_total": round(adjusted_total, 2),
                "projection_max": round(adjusted_max, 2),
                "safe_under_line": max(0.0, safe_under),
                "safe_over_line": safe_over,
                "model_confidence": round(combined_confidence, 2),
                "momentum_factor": round(momentum_factor, 2),
                "momentum_direction": "📈 Ascendente" if momentum_factor > 1.0 else "📉 Descendente" if momentum_factor < 1.0 else "➡️ Neutro"
            }
        
        return filtered_predictions
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences of shape (samples, sequence_length, 4)
            y_train: Training targets (momentum factors)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
        """
        if self.model:
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
    
    def save_model(self, path):
        """
        Save the trained LSTM model.
        
        Args:
            path: Path to save model
        """
        if self.model:
            self.model.save(path)
            self.model_path = path
    
    def load_model(self, path):
        """
        Load a pre-trained LSTM model.
        
        Args:
            path: Path to model file
        """
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
            self.model_path = path
