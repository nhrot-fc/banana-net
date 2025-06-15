# Roadmap de Banana-Net

Este documento detalla el plan de implementación y las tareas pendientes del proyecto Banana-Net para la detección y clasificación de vocalizaciones de primates.

## 1. Estado Actual ✅

### Estructura del Proyecto
- [x] Estructura básica de directorios creada (`src`, `data`, `notebooks`, `docs`)
- [x] Configuración de paquetes y dependencias (`pyproject.toml`)
- [x] Módulos principales inicializados

### Modelos de Datos
- [x] Enumeraciones para especies y tipos de llamada (`Specie`, `CallType`)
- [x] Clase de anotación para representar eventos acústicos (`Annotation`)

### Utilidades
- [x] Configuración de logging
- [x] Funciones básicas de carga de datos (`loading.py`)
- [x] Funciones básicas de preprocesamiento (`preprocessing.py`)

## 2. Próximos Pasos 🚧

### Preprocesamiento de Datos (Prioridad Alta)
- [ ] Completar funciones de exploración y limpieza de datos en `exploration.py`
- [ ] Implementar balanceo de clases y división de conjunto de datos (train/val/test)
- [ ] Implementar aumento de datos para audio (data augmentation)
- [ ] Completar la normalización de características

### Implementación de la Red YOLO (Prioridad Alta)
- [ ] Completar la función `create_yolov2_target` para convertir anotaciones a formato YOLO
- [ ] Desarrollar la arquitectura de la red (YOLOv2 adaptada para audio)
- [ ] Implementar funciones de pérdida personalizadas
- [ ] Configurar pipeline de entrenamiento

### Entrenamiento y Evaluación (Prioridad Media)
- [ ] Implementar métricas de evaluación (mAP, recall por clase)
- [ ] Configurar callbacks para guardar checkpoints y monitoreo
- [ ] Implementar visualización de resultados en tiempo real
- [ ] Configurar experimentos con diferentes hiperparámetros

### Inferencia (Prioridad Baja)
- [ ] Crear pipeline de inferencia para archivos de audio nuevos
- [ ] Optimizar modelo para inferencia (cuantización, podado)
- [ ] Implementar detección en tiempo real (si aplica)

### Documentación y Pruebas (Prioridad Media)
- [ ] Completar documentación de API para todos los módulos
- [ ] Escribir tests unitarios
- [ ] Crear ejemplos de uso y notebooks demostrativos
- [ ] Actualizar README con instrucciones detalladas

## 3. Detalles de Implementación Pendientes 🔍

### AudioYOLODataset
- [ ] Finalizar la conversión de anotaciones a formato YOLO
- [ ] Implementar normalización adecuada de anchors
- [ ] Añadir técnicas de data augmentation específicas para audio

### Transformaciones
- [ ] Implementar la lógica completa de YOLO en `create_yolov2_target`:
  - [ ] Asignación de anotaciones a celdas de la cuadrícula
  - [ ] Cálculo de IoU para selección del mejor anchor 
  - [ ] Codificación de dimensiones de cajas
  - [ ] One-hot encoding de clases

### Modelo
- [ ] Definir la arquitectura de la red adaptada para espectrogramas
- [ ] Implementar función de pérdida multitarea (localización, clasificación)
- [ ] Configurar estrategias para manejar el desbalanceo de clases

## 4. Timeline Estimado 📅

| Fase | Tiempo Estimado | Descripción |
|------|-----------------|-------------|
| Preprocesamiento | 2 semanas | Limpieza de datos, exploración, división de conjuntos |
| Desarrollo Modelo | 3 semanas | Implementación de YOLO para audio |
| Entrenamiento | 2 semanas | Experimentos con diferentes configuraciones |
| Evaluación | 1 semana | Análisis de resultados y métricas |
| Documentación | 1 semana | Finalizar documentación y ejemplos |

## 5. Problemas Conocidos 🐞
1. La normalización de anchors entre espacios absolutos y relativos necesita verificación
2. Posibles inconsistencias en los formatos de archivos de anotación
3. Falta integrar el logger en todos los módulos