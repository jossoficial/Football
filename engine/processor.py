"""Data processing utilities for feature extraction and preparation."""
import numpy as np

class DataProcessor:
    """Enhanced data processor with momentum-aware feature engineering."""
    
    def _value_or_none(self, value):
        """Convert value to float or None."""
        return value if isinstance(value, (int, float)) else None

    def _extract_stats(self, match, stats, team_id):
        """Extract individual match statistics for a team."""
        match = match or {}
        stats = stats or {}

        home_team = match.get("home_team") or {}
        away_team = match.get("away_team") or {}
        
        home_id = str(home_team.get("id"))
        away_id = str(away_team.get("id"))
        side = "home" if str(team_id) == home_id else "away"

        score = match.get("score") or {}
        overview = stats.get("overview") or {}

        def get_stat_value(stat_name):
            stat_obj = overview.get(stat_name) or {}
            all_obj = stat_obj.get("all") or {}
            return self._value_or_none(all_obj.get(side))

        return {
            "goals": self._value_or_none(score.get(side)),
            "corners": get_stat_value("corner_kicks"),
            "shots_on_target": get_stat_value("shots_on_target"),
            "total_shots": get_stat_value("total_shots")
        }

    def calculate_averages(self, history, team_id):
        """Calculate average statistics from match history."""
        if not history:
            return None

        metrics = ["goals", "corners", "shots_on_target", "total_shots"]
        sums = {m: 0.0 for m in metrics}
        counts = {m: 0 for m in metrics}

        for item in history:
            if not item:
                continue
            extracted = self._extract_stats(item.get("match"), item.get("stats"), team_id)
            for m in metrics:
                val = extracted.get(m)
                if val is not None:
                    sums[m] += val
                    counts[m] += 1

        averages = {}
        for m in metrics:
            averages[m] = sums[m] / counts[m] if counts[m] > 0 else None
        
        return averages
    
    def calculate_recent_averages(self, history, team_id, recent_matches=3):
        """Calculate averages for only recent matches (for momentum analysis)."""
        if not history or len(history) < recent_matches:
            return self.calculate_averages(history, team_id)
        
        recent_history = history[-recent_matches:]
        return self.calculate_averages(recent_history, team_id)

    def prepare_features(self, team_a_avg, team_b_avg):
        """Prepare combined feature vector from team averages."""
        combined = {}
        if not team_a_avg or not team_b_avg:
            return combined
            
        for key in team_a_avg.keys():
            if team_a_avg.get(key) is not None and team_b_avg.get(key) is not None:
                combined[f'{key}_a'] = team_a_avg[key]
                combined[f'{key}_b'] = team_b_avg[key]
                combined[f'{key}_max'] = max(team_a_avg[key], team_b_avg[key])
                combined[f'{key}_total'] = team_a_avg[key] + team_b_avg[key]
        return combined
    
    def enrich_features_with_momentum(self, features, history_a, history_b, team_a_id, team_b_id):
        """
        Enrich feature set with momentum indicators.
        
        Args:
            features: Base feature dictionary
            history_a: Match history for team A
            history_b: Match history for team B
            team_a_id: Team A ID
            team_b_id: Team B ID
            
        Returns:
            Dictionary: Features enhanced with momentum indicators
        """
        enriched = features.copy()
        
        # Calculate recent vs overall averages for momentum signals
        if history_a:
            recent_a = self.calculate_recent_averages(history_a, team_a_id, recent_matches=3)
            overall_a = self.calculate_averages(history_a, team_a_id)
            
            for metric in ['goals', 'corners', 'shots_on_target', 'total_shots']:
                if recent_a.get(metric) and overall_a.get(metric) and overall_a[metric] > 0:
                    momentum_ratio = recent_a[metric] / overall_a[metric]
                    enriched[f'{metric}_a_momentum'] = round(momentum_ratio, 2)
        
        if history_b:
            recent_b = self.calculate_recent_averages(history_b, team_b_id, recent_matches=3)
            overall_b = self.calculate_averages(history_b, team_b_id)
            
            for metric in ['goals', 'corners', 'shots_on_target', 'total_shots']:
                if recent_b.get(metric) and overall_b.get(metric) and overall_b[metric] > 0:
                    momentum_ratio = recent_b[metric] / overall_b[metric]
                    enriched[f'{metric}_b_momentum'] = round(momentum_ratio, 2)
        
        return enriched
