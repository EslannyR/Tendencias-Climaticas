# Tendencias Climáticas y Predicción Global de Anomalías de Temperatura

## Descripción del Proyecto

Análisis integral de las tendencias históricas de temperatura global y concentración de CO₂ atmosférico, utilizando técnicas de análisis de datos y ciencia de datos para comprender la evolución del cambio climático y desarrollar modelos predictivos.

## Objetivo

Analizar y predecir el comportamiento de la temperatura global en relación con la concentración de CO₂, mediante análisis exploratorio, modelado predictivo y visualización de datos, aportando evidencia cuantitativa sobre las tendencias del cambio climático.

## Estructura del Proyecto

```
clima_proyecto/
├── data/
│   ├── raw/                    # Datos originales
│   └── clean/                  # Datos procesados
├── notebooks/
│   └── 01_preparacion_datos_y_eda_completo.ipynb
├── outputs/                    # Resultados y visualizaciones
├── powerbi/                    # Dashboard interactivo
└── README.md
```

## Datasets

### 1. NOAA Global Land and Ocean Temperature Anomalies
- **Fuente**: National Centers for Environmental Information (NOAA)
- **Período**: 1958-2025 (68 años)
- **Variable**: Anomalía de temperatura global (°C) respecto al promedio 1901-2000
- **Acceso**: [NOAA Global Temperature Anomalies](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/tavg/1/8/1850-2025) - Julio 2025

### 2. NOAA Mauna Loa CO₂ Monthly Data
- **Fuente**: NOAA Earth System Research Laboratory
- **Período**: 1958-2023 (datos mensuales agregados anualmente)
- **Variable**: Concentración atmosférica de CO₂ (ppm)
- **Acceso**: [NOAA Mauna Loa CO₂](https://gml.noaa.gov/ccgg/trends/) - Julio 2025

## Metodología

### Fase 1: Preparación de Datos
- Limpieza y estandarización de datasets
- Unificación temporal (1958-2025)
- Creación de variables derivadas:
  - Tasas de cambio anuales
  - Aceleración (segunda derivada)
  - Tendencias móviles de 10 años
  - Agrupación por décadas

### Fase 2: Análisis Exploratorio (EDA)
- Estadísticas descriptivas y distribuciones
- Análisis de correlaciones temporales con rezagos
- Caracterización de regímenes climáticos históricos
- Análisis de volatilidad y puntos de inflexión
- Visualizaciones integrales (4 figuras temáticas)

### Fase 3: Modelado Predictivo *(En desarrollo)*
- Modelos de series temporales (ARIMA, Prophet)
- Regresiones multivariables CO₂ → Temperatura
- Evaluación con métricas RMSE y MAPE
- Proyecciones a 5 años

### Fase 4: Visualización Ejecutiva *(Planeada)*
- Dashboard interactivo en Power BI
- KPIs climáticos clave
- Comparaciones históricas y proyecciones

## Hallazgos Principales

### Correlaciones
- **CO₂ vs Temperatura**: 0.9565 (relación extremadamente fuerte)
- **Correlación temporal óptima**: Sin rezagos significativos (sincronización inmediata)

### Regímenes Climáticos
| Período | Pendiente Temp. (°C/año) | Pendiente CO₂ (ppm/año) |
|---------|-------------------------|------------------------|
| Pre-industrial (1958-1970) | -0.0035 | 0.83 |
| Industrialización (1970-1990) | 0.0159 | 1.48 |
| Aceleración (1990-2010) | 0.0135 | 1.83 |
| Crisis climática (2010-2023) | **0.0262** | **2.44** |

### Insights Clave
- **Aceleración del cambio**: Factor de 2.5x en tasas de cambio de temperatura reciente vs histórica
- **Régimen más crítico**: 2010-2023 con mayor pendiente de calentamiento registrada
- **Aceleración sostenida**: Tendencia consistentemente creciente desde 2000

## Tecnologías Utilizadas

### Análisis y Procesamiento
- **Python 3.11**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **SciPy**: Análisis estadístico avanzado

### Visualización
- **Matplotlib**: Gráficos base
- **Seaborn**: Visualizaciones estadísticas
- **Power BI**: Dashboard ejecutivo *(en desarrollo)*

### Entorno
- **Jupyter Notebooks**: Desarrollo interactivo
- **Git**: Control de versiones

## Instalación y Uso

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Ejecución
1. Clona el repositorio
2. Instala las dependencias
3. Ejecuta `01_preparacion_datos_y_eda_completo.ipynb`

## Resultados

### Dataset Procesado
- **68 observaciones** × **11 variables**
- **Período**: 1958-2025
- **Variables derivadas**: Cambios, aceleración, tendencias móviles
- **Correlación validada**: 0.9565 CO₂-Temperatura

### Visualizaciones Generadas
1. **Evolución Temporal**: Series históricas y comparación conjunta
2. **Análisis Estadístico**: Distribuciones y tendencias por década
3. **Análisis Avanzado**: Cambios, aceleración y volatilidad
4. **Correlaciones**: Matriz y análisis por regímenes climáticos

## Próximos Pasos

- [ ] Implementar modelos ARIMA y Prophet
- [ ] Desarrollar dashboard interactivo en Power BI
- [ ] Generar proyecciones climáticas a 5 años
- [ ] Documentar metodología para replicabilidad

## Contribuciones

Este proyecto forma parte de un portafolio de ciencia de datos enfocado en análisis climático y modelado predictivo.

## Contacto

**Eslanny Ramírez** - Analista de Datos
- GitHub: [\[@EslannyR\]](https://github.com/EslannyR)
- LinkedIn: [\[@EslannyRamirez\]](https://www.linkedin.com/in/eslannyramirez/)

---

*Proyecto desarrollado con fines académicos y de investigación aplicada en ciencia de datos climática.*