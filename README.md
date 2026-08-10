# Football - Predicción de Partidos con ML Avanzado

[![Release](https://img.shields.io/github/v/release/jossjb865/Football?label=release)](https://github.com/jossjb865/Football/releases)
[![License](https://img.shields.io/github/license/jossjb865/Football)](https://github.com/jossjb865/Football/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/jossjb865/Football)](https://github.com/jossjb865/Football/issues)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![ML Models](https://img.shields.io/badge/models-CatBoost%20%26%20LSTM-brightgreen)]()

## 📋 Descripción

Football es una aplicación avanzada de predicción de partidos que utiliza **CatBoost** para modelado predictivo y **LSTM** para análisis de momentum. Genera proyecciones precisas de mercados de apuestas (goles, tiros, corners) basadas en datos históricos y tendencias de rendimiento en tiempo real.

## ✨ Características

- 🤖 **Modelo CatBoost**: Regresión robusta con validación automática de características
- 📈 **Filtro LSTM Momentum**: Análisis temporal de 5 partidos recientes para capturar tendencias
- 🔄 **Pipeline Integrado**: Flujo completo desde obtención de datos hasta predicción final
- 📊 **Enriquecimiento de Features**: Indicadores de momentum para mayor precisión
- 🎯 **Líneas Seguras**: Cálculo profesional de márgenes de apuesta
- 📋 **Confianza Modelada**: Métricas de confianza ajustadas por momentum
- 🌐 **API Integration**: Conexión con TheStatsAPI para datos en tiempo real

## 🏗️ Arquitectura

```
Football/
├── main.py                          # Pipeline principal
├── engine/
│   ├── __init__.py                 # Exportaciones del paquete
│   ├── api_client.py               # Cliente TheStatsAPI (10 últimos partidos)
│   ├── processor.py                # Procesamiento de datos + momentum
│   └── models/
│       ├── __init__.py             # Exportaciones de modelos
│       ├── catboost_model.py       # Modelo CatBoost para predicción
│       └── lstm_momentum.py        # LSTM para análisis de momentum
├── requirements.txt                # Dependencias
└── README.md
```

## 🛠️ Instalación

### Requisitos
- Python >= 3.8
- pip o conda

### Setup Local

```bash
# Clonar repositorio
git clone https://github.com/jossoficial/Football.git
cd Football

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
export THESTATSAPI_KEY=tu_clave_aqui
export ISPORTSAPI_KEY=tu_clave_aqui
```

## 🚀 Uso

### Ejecución Básica

```bash
# Ejecutar predicción para un partido específico
MATCH_ID=123456 python main.py
```

### Output Esperado

```
--- Iniciando Proyección Automática para Match ID: 123456 ---

[1/5] Obteniendo detalles del partido...
✅ Partido: Manchester United vs Liverpool (2026-08-15)

[2/5] Recopilando estadísticas históricas detalladas...
[3/5] Procesando 10 partidos de Manchester United y 10 de Liverpool...
[4/5] Ejecutando modelo CatBoost...
   📊 Momentum Manchester United: 1.05
   📊 Momentum Liverpool: 0.98
   📊 Momentum combinado: 1.02

[5/5] Aplicando filtro de momentum...

============================================================
 🎯 PROYECCIÓN FINAL: Manchester United vs Liverpool
 📅 Fecha: 2026-08-15
 🔄 Momentum: 1.02 ↗️ Ascendente
============================================================

📈 Mercado: GOALS
   ├─ Proyección Total: 2.85
   ├─ Proyección Máximo: 1.95
   ├─ LÍNEA BAJO (UNDER): < 1.00
   ├─ LÍNEA ALTO (OVER): > 4.50
   ├─ Confianza del modelo: 85%
   └─ Factor de momentum: 1.02
```

## 📊 Pipeline de Predicción

### 1. Obtención de Datos
- API TheStatsAPI para detalles del partido
- Historial de últimos 10 partidos por equipo
- Estadísticas detalladas (goles, tiros, corners, etc.)

### 2. Procesamiento
- Extracción de métricas por posición (local/visitante)
- Cálculo de promedios históricos
- Preparación de vector de características (16 features)
- Enriquecimiento con indicadores de momentum (4 adicionales)

### 3. Predicción CatBoost
- Validación automática de características
- Regresión RMSE con profundidad 6
- Fallback a promedio si datos insuficientes
- Confianza del modelo: 0.8

### 4. Filtro LSTM Momentum
- Secuencia de últimos 5 partidos
- Normalización de valores [0,1]
- Red LSTM (64→32 neuronas)
- Cálculo de factor de momentum (0.8-1.2)
- Ajuste de confianza combinada

### 5. Resultado Final
- Predicción ajustada por momentum
- Líneas de apuesta profesionales
- Confianza del modelo: 0.5-0.95
- Dirección de momentum (📈/📉/→)

## 🔧 Configuración

### Variables de Entorno (.env)

```bash
THESTATSAPI_KEY=tu_api_key_aqui
ISPORTSAPI_KEY=tu_api_key_aqui
MATCH_ID=123456
```

### Parámetros del Modelo

**CatBoost**:
```python
iterations=100              # Número de iteraciones
learning_rate=0.1         # Tasa de aprendizaje
depth=6                   # Profundidad del árbol
loss_function='RMSE'      # Función de pérdida
```

**LSTM Momentum**:
```python
sequence_length=5         # Partidos a analizar
hidden_units=[64, 32]     # Capas LSTM
epochs=50                 # Épocas de entrenamiento
```

## 📈 Métricas de Salida

### Por Mercado (goles, tiros, corners)
- **projection_total**: Suma esperada de ambos equipos
- **projection_max**: Valor máximo esperado por equipo
- **safe_under_line**: Línea conservadora bajo (UNDER)
- **safe_over_line**: Línea conservadora sobre (OVER)
- **model_confidence**: Confianza 0.0-1.0 (ajustada por momentum)
- **momentum_factor**: Multiplicador de momentum (0.8-1.2)
- **momentum_direction**: Indicador visual de tendencia

## 🧪 Entrenamiento de Modelos

### Entrenar CatBoost

```python
from engine.models import CatBoostPredictor

model = CatBoostPredictor()
model.train(X_train, y_train)
model.save_model('models/catboost_model.pkl')
```

### Entrenar LSTM

```python
from engine.models import LSTMMomentumFilter

lstm = LSTMMomentumFilter(sequence_length=5)
lstm.train(X_train, y_train, X_val, y_val, epochs=50)
lstm.save_model('models/lstm_momentum.h5')
```

## 🐛 Troubleshooting

### Error: "429 Client Error: Too Many Requests"
- La API está bloqueando por rate limiting
- Aumentar `time.sleep()` en `api_client.py`
- Verificar límites de API key

### Error: "No se pudieron calcular promedios"
- El equipo no tiene suficiente historial
- Requiere mínimo 1 partido en el historial
- Verificar que el MATCH_ID sea válido

### Advertencia: "Características incompletas"
- Datos de algunas métricas no disponibles
- El modelo usa valores por defecto (0.0)
- Confianza reducida a 0.5 (fallback)

## 📚 Dependencias

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
requests>=2.26.0
joblib>=1.1.0
catboost>=1.0.0
tensorflow>=2.10.0
keras>=2.10.0
```

## 🤝 Cómo Contribuir

1. Fork el repositorio
2. Crea rama: `git checkout -b feat/mejora`
3. Commit: `git commit -m "feat: descripción"`
4. Push: `git push origin feat/mejora`
5. Abre Pull Request

## 📝 Roadmap

- [ ] Dashboard web con Streamlit
- [ ] Historial de predicciones y accuracy
- [ ] Modelos por liga/competición
- [ ] Exportación de reportes PDF
- [ ] API REST para integraciones
- [ ] Backtesting automático
- [ ] Integración con múltiples APIs

## ⚖️ Licencia

MIT License - Ver [LICENSE](LICENSE)

## 📞 Contacto

**Mantenedor**: jossoficial  
**Email**: jossoficial78@gmail.com  
**GitHub**: [@jossoficial](https://github.com/jossoficial)

---

**⚠️ Descargo de Responsabilidad**  
Este proyecto es para análisis educativo. Úsalo responsablemente en apuestas deportivas. No garantizamos exactitud de predicciones.
