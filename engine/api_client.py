import requests
import os
import time
from datetime import datetime

class FootballDataClient:
    def __init__(self):
        self.stats_api_key = os.getenv('THESTATSAPI_KEY')
        self.isports_api_key = os.getenv('ISPORTSAPI_KEY')
        self.base_url = "https://api.thestatsapi.com/api"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.stats_api_key}"})

    def get(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 429:
                raise requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error en llamada a {endpoint}: {e}")
            return None

    def get_match_details(self, match_id):
        return self.get(f"/football/matches/{match_id}")

    def get_matches_by_date(self, date_str):
        """
        Obtiene los partidos para una fecha específica.
        Recorre todas las páginas si es necesario y filtra por fecha exacta.
        
        Args:
            date_str: Fecha en formato YYYY-MM-DD
            
        Returns:
            Lista de partidos para la fecha especificada
        """
        all_matches = []
        page = 1
        
        print(f"[API] Solicitando partidos para la fecha: {date_str}")
        
        while True:
            params = {
                "date_from": date_str,
                "date_to": date_str,
                "per_page": 100,
                "page": page
            }
            
            response = self.get("/football/matches", params=params)
            
            if not response:
                break
            
            matches = response.get("data", [])
            
            # Filtrar partidos por fecha exacta para evitar partidos de otros días
            filtered_matches = self._filter_matches_by_date(matches, date_str)
            all_matches.extend(filtered_matches)
            
            # Verificar si hay más páginas
            meta = response.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            
            print(f"[API] Página {page}/{total_pages} - {len(filtered_matches)} partidos (filtrados)")
            
            if page >= total_pages:
                break
            
            page += 1
            # Pausa de cortesía entre páginas
            time.sleep(0.5)
        
        print(f"[API] Se encontraron {len(all_matches)} partidos en total")
        
        return all_matches

    def _filter_matches_by_date(self, matches, target_date):
        """
        Filtra los partidos para asegurar que solo devuelve los de la fecha exacta.
        
        Args:
            matches: Lista de partidos
            target_date: Fecha objetivo en formato YYYY-MM-DD
            
        Returns:
            Lista filtrada de partidos
        """
        filtered = []
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        for match in matches:
            # Obtener la fecha del partido de diferentes formatos posibles
            utc_date = match.get("utc_date") or match.get("date")
            
            if utc_date:
                try:
                    # Parsear la fecha (puede estar en formato ISO con timestamp)
                    if "T" in utc_date:
                        match_date_obj = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date()
                    else:
                        match_date_obj = datetime.strptime(utc_date, "%Y-%m-%d").date()
                    
                    # Solo incluir si la fecha coincide exactamente
                    if match_date_obj == target_date_obj:
                        filtered.append(match)
                except ValueError:
                    # Si no se puede parsear la fecha, incluir el partido
                    filtered.append(match)
        
        return filtered

    def get_historical_team_data(self, team_id, match_date):
        """
        Implementación optimizada: Procesa máximo 10 partidos recientes.
        """
        params = {
            "team_id": str(team_id),
            "date_to": match_date,
            "status": "finished",
            "per_page": 10, # Solicitamos 10 a la API
            "utc_offset": "-06:00"
        }

        print(f"\n[API] Solicitando historial para Team ID: {team_id}")
        response = self.get("/football/matches", params=params)

        if not response:
            return []

        matches = response.get("data", [])

        # Ordenar del más reciente al más antiguo para asegurar el momentum
        matches.sort(
            key=lambda match: match.get("utc_date", ""),
            reverse=True
        )

        # Garantizar que no procesamos más de 10
        matches = matches[:10]
        print(f"[API] Procesando exactamente {len(matches)} partidos encontrados...")

        history = []
        for index, match in enumerate(matches, start=1):
            match_id = match.get("id")
            if not match_id:
                continue

            # Pausa de cortesía para evitar el error 429
            time.sleep(0.5)

            stats_data = self._get_stats_with_retry(match_id)
            if stats_data:
                history.append({
                    "match": match,
                    "stats": stats_data
                })
            
            print(f"   > Procesado {index}/{len(matches)}")

        return history

    def _get_stats_with_retry(self, match_id, retries=2, backoff=3):
        """Maneja reintentos si la API nos bloquea temporalmente."""
        for i in range(retries):
            try:
                stats_response = self.get(f"/football/matches/{match_id}/stats")
                if stats_response and stats_response.get("data"):
                    return stats_response["data"]
                return None
            except Exception as e:
                if "429" in str(e):
                    wait = backoff * (i + 1)
                    print(f"⚠️ Rate Limit detectado. Reintentando en {wait}s...")
                    time.sleep(wait)
                else:
                    return None
        return None
