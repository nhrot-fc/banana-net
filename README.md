# banana-net

Proyecto para detección y clasificación de sonidos de primates mediante redes neuronales.

## Estructura del proyecto

El proyecto se organiza de la siguiente manera:

```
banana-net/
├── data/                   # Datos del proyecto
│   ├── processed/          # Datos procesados
│   └── raw/                # Datos brutos
├── docs/                   # Documentación
├── logs/                   # Logs del proyecto
├── notebooks/              # Jupyter notebooks
├── scripts/                # Scripts de utilidad
├── src/                    # Código fuente principal
│   └── banana_net/         # Paquete principal
│       ├── models/         # Modelos y estructuras de datos
│       ├── testing/        # Utilidades para pruebas
│       ├── training/       # Código para entrenamiento de modelos
│       └── utils/          # Utilidades generales
│           ├── audio/      # Procesamiento de audio
│           ├── spectrogram/# Procesamiento de espectrogramas
│           └── yolo/       # Utilidades para YOLO
└── tests/                  # Pruebas unitarias (sigue la estructura de src/)
    └── banana_net/
        └── utils/
            ├── audio/
            ├── spectrogram/
            └── yolo/

## Instalación

### Para desarrollo

```bash
# Instalar en modo desarrollo
make dev-install

# Instalar dependencias de desarrollo
make install-dev-deps
```

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
# Ejecutar todas las pruebas
make test

# Ejecutar pruebas con reporte de cobertura
make test-cov
```

## Procesamiento y Estandarización de Datos

El dataset de audios y anotaciones de primates pasa por un riguroso proceso de estandarización y limpieza para asegurar la calidad de los datos que alimentarán a los modelos de machine learning. Este proceso se documenta y ejecuta en el notebook `notebooks/exploration.ipynb` y utiliza funciones de los módulos en `src/banana_net/utils/`.

A continuación, se detallan los pasos clave del protocolo:

### 1. Carga de Datos

El primer paso consiste en cargar todas las anotaciones y datos de audio desde los directorios de especies.

- **`load_audio_and_annotations_from_species_dirs(raw_data_dir, species_dirs)`** en `src/banana_net/utils/loading.py`: Esta es la función principal de carga. Itera sobre los directorios de especies, empareja los archivos de anotaciones (`.txt`) con sus correspondientes archivos de audio (`.wav`), y carga su contenido. Los pares de archivos que presentan errores durante la carga (ya sea en el audio o en la anotación) son descartados para mantener la integridad del dataset.

- **`load_single_annotation_file(file_path)`** en `src/banana_net/utils/loading.py`: Carga un único archivo de anotaciones de Raven (`.txt`). Estandariza los nombres de las columnas (p. ej., "Begin Time (s)" a "begin_time") y añade dos columnas importantes: `recording_file` (el nombre del archivo `.wav` asociado) y `directory` (el nombre del directorio de la especie de donde proviene el archivo).

### 2. Validación y Limpieza Inicial

Durante la carga y el análisis exploratorio en el notebook, se realizan varias validaciones para asegurar la calidad de los datos:

- **Consistencia de archivos**: Se verifica que cada archivo de anotaciones `.txt` tenga un archivo de audio `.wav` con el mismo nombre base.
- **Duración del audio**: Se comprueba que los tiempos de fin de las anotaciones (`end_time`) no excedan la duración total del archivo de audio correspondiente. Esto previene errores en el procesamiento posterior.
- **Análisis de Espectrogramas**: Se calculan los espectrogramas para cada archivo de audio y se emiten advertencias si contienen valores de energía cero. Los valores cero pueden causar problemas matemáticos (p. ej., `log(0)`) en los pasos de extracción de características.

### 3. Estandarización de Anotaciones

Una vez cargados los datos, se aplica una serie de funciones para limpiar y normalizar las etiquetas y los valores.

- **`clean_annotation_dataframe(df)`** en `src/banana_net/utils/preprocessing.py`: Realiza una limpieza exhaustiva del DataFrame de anotaciones. Elimina filas con valores nulos en columnas clave, convierte columnas a tipos numéricos, y filtra entradas no deseadas como aquellas que contienen "noise" o caracteres no alfabéticos en las columnas `specie` y `call_type`.

- **`normalize_labels(df)`** en `src/banana_net/utils/preprocessing.py`: Estandariza las etiquetas de `specie` y `call_type`. Convierte todo a minúsculas y aplica un diccionario de correcciones para unificar etiquetas que se refieren a lo mismo (p. ej., "contact call" y "contact syllable" se convierten en "cc" y "cs" respectivamente).

- **`find_potential_typos_per_species(df, known_species)`** en `src/banana_net/utils/data_preprocessing.py`: Utiliza la **distancia de Levenshtein** para detectar posibles errores tipográficos en las etiquetas. Compara los valores poco comunes con una lista de valores conocidos y sugiere correcciones, ayudando a mantener la consistencia del dataset.

### 4. Creación de Características

Se generan nuevas características a partir de las anotaciones básicas para enriquecer el dataset.

- **`create_feature_set(df)`** en `src/banana_net/utils/preprocessing.py`: Calcula dos características fundamentales para el análisis de las vocalizaciones:
    - `duration_s`: La duración de la llamada en segundos (`end_time` - `begin_time`).
    - `bandwidth_hz`: El ancho de banda de la llamada en Hertz (`high_freq` - `low_freq`).

### 5. Preprocesamiento Avanzado y Consistencia

Se aplican reglas de negocio y verificaciones adicionales para refinar el dataset.

- **`preprocess_annotations(df)`** en `src/banana_net/utils/data_preprocessing.py`: Esta función aplica transformaciones adicionales. Aunque su nombre es genérico, en la implementación actual se centra en recalcular `duration_s` y `bandwidth_hz` para asegurar que estén presentes antes de los siguientes pasos.

- **`check_species_directory_consistency(df)`** y **`fix_species_directory_consistency(df)`** en `src/banana_net/utils/preprocessing.py`: Estas funciones aseguran la integridad referencial de los datos. Extraen el código de la especie del nombre del directorio (p. ej., "SB" de `bolivian_squirrel_monkey__SB`) y lo comparan con el valor en la columna `specie`. `check_...` reporta las inconsistencias, mientras que `fix_...` las corrige, asegurando que la especie de la anotación coincida con su directorio de origen.

### 6. Cálculo de Anchor Boxes para Detección de Objetos

Para preparar los datos para un modelo de detección de objetos como YOLO, es crucial definir "anchor boxes" que representen los tamaños y formas típicos de las llamadas en los espectrogramas.

- **`extract_dimensions_by_group(processed_df)`** en `src/banana_net/utils/data_preprocessing.py`: Agrupa todas las anotaciones por especie y tipo de llamada. Para cada grupo, extrae las dimensiones (`duration_s` y `bandwidth_hz`) en un array de NumPy.

- **`calculate_anchor_boxes(dimensions_dict, k)`** (definida en el notebook `exploration.ipynb`): Toma el diccionario de dimensiones y aplica el algoritmo de clustering **k-means** a cada grupo. Los centroides de los clusters resultantes se convierten en los anchor boxes, que son las dimensiones promedio de las vocalizaciones. Esto se hace tanto para grupos específicos (p. ej., especie + tipo de llamada) como para el dataset completo para obtener anchor boxes globales.

### 7. Exportación del Dataset Procesado

- Finalmente, el `DataFrame` de `pandas`, que ahora contiene todas las anotaciones procesadas, limpias, y enriquecidas con nuevas características, se exporta a un único archivo CSV: `data/processed/processed_dataset.csv`.
- Este archivo consolidado sirve como la entrada final y estandarizada para los scripts de entrenamiento de los modelos de deep learning.

Este protocolo detallado asegura que los datos sean consistentes, limpios y estén optimizados para el entrenamiento de modelos de alto rendimiento.
