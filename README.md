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
