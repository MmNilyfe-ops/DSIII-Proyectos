# Pre-entrega Data Science III — Clasificador de noticias con PyTorch

## Descripción

Este proyecto implementa un pipeline base de Deep Learning para clasificar noticias del dataset **AG News** en cuatro categorías:

- World
- Sports
- Business
- Sci/Tech

El objetivo principal de esta pre-entrega es construir una infraestructura reproducible de entrenamiento y validación en PyTorch, priorizando la calidad del pipeline por encima de la complejidad del modelo.

## Objetivo del proyecto

Desarrollar un clasificador de texto multiclase capaz de recibir una noticia y asignarla a una de las cuatro categorías disponibles.

El pipeline incluye:

- carga de datos desde Google Drive;
- limpieza y combinación de título y descripción;
- separación en entrenamiento, validación y prueba;
- tokenización;
- construcción de vocabulario;
- conversión de textos a tensores;
- creación de `Dataset` y `DataLoader`;
- definición de un modelo con `nn.Module`;
- entrenamiento con Adam;
- validación por época;
- registro de pérdida y accuracy;
- guardado del mejor checkpoint;
- evaluación final sobre datos no vistos;
- predicción sobre textos nuevos.

## Dataset

Se utiliza el dataset **AG News Classification Dataset**, disponible en Kaggle:

https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

Los archivos utilizados son:

```text
train.csv
test.csv
```

Cada archivo contiene las columnas:

- `Class Index`
- `Title`
- `Description`

Las clases originales están numeradas del 1 al 4 y se convierten a valores del 0 al 3 para utilizar `CrossEntropyLoss` en PyTorch.

### Clases

| Etiqueta | Categoría |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## Carga desde Google Drive

Los CSV no se incluyen en este repositorio debido a su tamaño.

Deben guardarse en Google Drive con la siguiente estructura:

```text
MyDrive/
└── CoderHouse/
    └── Data Science III/
        ├── train.csv
        └── test.csv
```

El notebook monta Google Drive mediante:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Luego carga los archivos desde:

```text
/content/drive/MyDrive/CoderHouse/Data Science III/train.csv
/content/drive/MyDrive/CoderHouse/Data Science III/test.csv
```

## Arquitectura del modelo

El clasificador base utiliza:

- `nn.EmbeddingBag`
- capa lineal
- activación `ReLU`
- `Dropout`
- capa de salida multiclase

La arquitectura resume cada noticia mediante el promedio de sus embeddings y luego procesa esa representación con una red neuronal simple.

## Configuración del entrenamiento

- Framework: PyTorch
- Optimizador: Adam
- Learning rate: `0.001`
- Batch size: `128`
- Número de épocas: `8`
- Semilla: `42`
- Función de pérdida: `CrossEntropyLoss`
- Métrica principal: Accuracy

El dispositivo se detecta automáticamente:

- CUDA, si hay una GPU NVIDIA disponible;
- MPS, si se utiliza Apple Silicon;
- CPU, en cualquier otro caso.

## Ciclo de entrenamiento

Cada iteración implementa explícitamente:

1. movimiento de tensores al dispositivo;
2. `optimizer.zero_grad()`;
3. forward pass;
4. cálculo de la pérdida;
5. `loss.backward()`;
6. `optimizer.step()`.

También se ejecuta una fase de validación separada utilizando `torch.no_grad()`.

## Resultados observados

Durante el entrenamiento, la pérdida de entrenamiento disminuyó de forma sostenida.

La pérdida de validación alcanzó su mejor valor alrededor de la época 4 y luego comenzó a aumentar levemente, mientras la pérdida de entrenamiento continuó bajando.

Esto indica un sobreajuste moderado a partir de la época 5.

Por este motivo, el pipeline guarda automáticamente el mejor checkpoint según la menor pérdida de validación.

## Guardado del modelo

El notebook guarda:

```text
best_agnews_model.pt
```

También puede generar:

```text
artifacts/
├── agnews_text_classifier.pt
├── vocab.json
└── training_history.json
```

Estos archivos no se incluyen en GitHub porque pueden regenerarse al ejecutar el notebook.

## Estructura del repositorio

```text
DSIII-Proyectos/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── PreEntregaDSIII-Martinelli.ipynb

```

## Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

Dependencias principales:

- pandas
- numpy
- matplotlib
- scikit-learn
- torch
- jupyter

## Ejecución en Google Colab

1. Abrir el notebook en Google Colab.
2. Montar Google Drive.
3. Verificar que `train.csv` y `test.csv` estén en la carpeta correcta.
4. Seleccionar GPU T4 si está disponible.
5. Ejecutar:

```text
Entorno de ejecución → Reiniciar sesión y ejecutar todo
```

6. Verificar que no haya errores.
7. Revisar las curvas de pérdida y accuracy.
8. Confirmar que se haya guardado el mejor checkpoint.

## Reproducibilidad

El proyecto utiliza semillas aleatorias fijas:

```python
SEED = 42
```

Esto permite reproducir:

- división de datos;
- orden de los lotes;
- inicialización de pesos;
- entrenamiento;
- resultados generales.

## Limitaciones

- El modelo utiliza una tokenización simple.
- No emplea embeddings preentrenados.
- `EmbeddingBag` no modela completamente el orden de las palabras.
- Se trabaja con una muestra del dataset para reducir el tiempo de entrenamiento.
- El rendimiento puede mejorar con LSTM, GRU o Transformers.
- No se implementó todavía una API de despliegue.

## Próximas mejoras

- agregar early stopping;
- probar embeddings preentrenados;
- entrenar una LSTM o GRU;
- comparar con un Transformer;
- agregar pruebas automatizadas;
- implementar una API simple;
- serializar y versionar los artefactos del modelo.

## Autora

**Mariana Martinelli**

Pre-entrega de Data Science III.
