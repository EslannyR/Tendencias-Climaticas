🌍 Análisis del Cambio Climático: CO₂ y Temperatura Global
📌 Objetivo del Proyecto

Este proyecto busca analizar la relación entre las concentraciones de dióxido de carbono (CO₂) en la atmósfera y la anomalía de la temperatura global en el período comprendido entre 1958 y 2023. El objetivo es explorar visualmente y estadísticamente cómo han evolucionado estas dos variables clave del cambio climático, identificar patrones por décadas, y establecer posibles correlaciones entre ambas.

📂 Fuentes de Datos
1. CO₂ Atmosférico (1958–2023)

Fuente: NOAA Earth System Research Laboratory

Ubicación de medición: Estación de Mauna Loa, Hawái

Formato: CSV con columnas de año, mes y concentración promedio mensual (ppm)

Acceso: https://gml.noaa.gov/ccgg/trends/

2. Anomalía de Temperatura Global (1880–2025)

Fuente: NOAA National Centers for Environmental Information (NCEI)

Descripción: Anomalía mensual en °C respecto al promedio del siglo XX

Formato: CSV con columnas de año, mes y anomalía

Acceso: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies

🔎 Para el análisis conjunto, se utilizaron únicamente los años coincidentes entre ambos datasets: 1958–2023.
🧪 Fase: Análisis Exploratorio de Datos (EDA)

Se llevaron a cabo los siguientes análisis:

📈 Visualización temporal de CO₂ y temperatura global (por año)

📊 Gráfico combinado con doble eje y para observar simultáneamente las dos variables

🧮 Promedios por década (1950s–2020s)

📦 Boxplots por década para observar la distribución y posibles valores atípicos

🔥 Mapa de calor de correlación entre variables (year, anomaly, co2_avg, decade)

🚫 Detección de outliers usando el rango intercuartílico (IQR), sin detección relevante

✅ Herramientas Utilizadas

Python (Jupyter Notebook)

pandas, numpy para manipulación de datos

matplotlib, seaborn para visualización

Power BI (fase futura): se utilizará para la construcción del dashboard final