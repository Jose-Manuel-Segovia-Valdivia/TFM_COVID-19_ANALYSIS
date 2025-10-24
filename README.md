# Análisis Estratégico de la Pandemia COVID-19
**Un enfoque basado en datos para la Identificación de Patrones y Factores Clave**

**Autor:** José Manuel Segovia Valdivia
**Máster:** Máster en Data Science y Big Data
**Universidad:** Universidad de Sevilla
**Fecha:** Octubre 2025

---

## 1. Resumen del Proyecto

Este repositorio contiene el código y los informes del Trabajo de Fin de Máster (TFM) que analiza los datos globales de la pandemia de COVID-19 (2020-2024). El objetivo principal es transformar el masivo y heterogéneo conjunto de datos de la pandemia en conocimiento accionable, identificando patrones de respuesta, factores de riesgo y métricas de eficiencia en la gestión sanitaria a nivel mundial.

El análisis completo está disponible en dos formatos en la carpeta `/Reports`:
* **Memoria Técnica (Analistas):** Un informe detallado que describe la metodología, el procesamiento de datos y los resultados estadísticos completos.
* **Memoria Ejecutiva (Sponsors):** Un resumen de alto nivel centrado en los hallazgos clave y las recomendaciones estratégicas.

## 2. Hallazgos Clave 🎯

El análisis ha permitido alcanzar varios logros fundamentales:

* **Identificación de Perfiles de Países:** Se han segmentado 235 países en **2 grupos distintos** (clústeres) basados en sus indicadores pandémicos y socioeconómicos. Este análisis ha revelado "arquetipos" de respuesta a la pandemia, permitiendo agrupar países con índices de respuesta similar.
* **Creación de un Ranking de Eficiencia Sanitaria:** Se ha desarrollado un índice que evalúa la gestión de la mortalidad de cada nación en relación con lo que se esperaría dadas sus características (PIB, demografía, etc.). Sorprendentemente, el ranking revela que muchas de las **grandes economías de Europa Occidental y Norteamérica tuvieron un rendimiento inferior a su potencial**.
* **Cuantificación de Factores de Riesgo y Protección:** Se ha cuantificado el impacto de factores clave, descubriendo una **interacción significativa entre la vacunación y la edad media**: el efecto protector de la vacunación fue notablemente superior en poblaciones más envejecidas.
* **Detección de Anomalías en los Datos:** Se ha implementado un sistema para identificar y visualizar automáticamente semanas con picos o caídas de datos anómalos en los reportes de casos y muertes, un paso crucial para evaluar la calidad del dato.

## 3. Estructura del Repositorio

El proyecto está estructurado de la siguiente manera:
```
TFM_Project/
├── .gitignore <-- Archivo para ignorar datos y caché
├── README.md <-- Esta página de presentación
├── Requirements.txt <-- Lista de librerías de Python
├── Reports/
    │ ├── TFM_analistas_Jose_Manuel_Segovia_Valdivia.pdf
    │ └── TFM_ejecutivos_Jose_Manuel_Segovia_Valdivia.pdf
├── Scripts/
    │ ├── 01_data_preparation.py
    │ ├── 02_exploratory_analysis.py
    │ ├── 03_clustering_analysis.py
    │ └── 04_advanced_modeling.py
└── Data/
    | ├── inputs/
        | │ ├── countries.geo.json
        │ └── .gitkeep (El dataset .csv se ignora con .gitignore)
    │ └── outputs/
        | └── .gitkeep (Esta carpeta se ignora con .gitignore)
```
## 4. Cómo Ejecutar el Proyecto

Este proyecto está diseñado como un pipeline secuencial. Los scripts deben ejecutarse en orden numérico, ya que la salida de uno es la entrada del siguiente.

### Paso 1: Configuración del Entorno

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/](https://github.com/)[TU_USUARIO]/[NOMBRE_DEL_REPOSITORIO].git
    cd [NOMBRE_DEL_REPOSITORIO]
    ```
2.  **(Recomendado) Crear un entorno virtual:**
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows: env\Scripts\activate
    ```
3.  **Instalar las dependencias:**
    ```bash
    pip install -r Requirements.txt
    ```

### Paso 2: Obtener los Datos de Entrada

Este repositorio no incluye el archivo `dataset_covid.csv` (700MB+) debido a las limitaciones de tamaño de GitHub. Para ejecutar el proyecto, es necesario descargarlo manualmente:

1.  **Descarga el dataset** desde la fuente original de Our World in Data:
    * **Enlace de descarga:** [OWID COVID-19 Data (Compact CSV)](https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv)
2.  **Renombra el archivo** descargado a: `dataset_covid.csv`
3.  **Coloca el archivo** dentro de la carpeta `Data/inputs/`.

La estructura de tu carpeta `Data/inputs/` debe quedar así:
Data/ └── inputs/ ├── countries.geo.json └── dataset_covid.csv <-- (El archivo que acabas de descargar)


### Paso 3: Ejecutar el Pipeline

Ejecuta los scripts en orden desde la carpeta raíz del proyecto (`TFM_Project/`).

```bash
python Scripts/01_data_preparation.py
python Scripts/02_exploratory_analysis.py
python Scripts/03_clustering_analysis.py
python Scripts/04_advanced_modeling.py
```
Al finalizar la ejecución, la carpeta Data/outputs/ (que está ignorada por Git) contendrá todos los datasets procesados, tablas y gráficos generados por el análisis.

