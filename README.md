# Conformal Prediction

Exposición del curso **Aprendizaje Automático (SI7009)**, Maestría en Ciencia de Datos y Analítica, Universidad EAFIT, 2026-2.

**Expositor:** Pedro Villegas Michel

## La idea en una frase

Un modelo normal entrega un solo número o una sola etiqueta. *Conformal prediction* le agrega un **rango o conjunto con una promesa de cobertura**: por ejemplo, "el valor real cae dentro del rango el 90 % de las veces".

## Los tres métodos

| Método | Qué hace | Ejemplo en este repositorio |
|---|---|---|
| 1. Split conformal | Una banda del mismo ancho para todos | Regresión con `XGBRegressor` |
| 2. CQR | Una banda que se ensancha donde hay más ruido | Regresión cuantílica con XGBoost |
| 3. Mondrian | Un umbral distinto para cada clase | Detección de fraude con `XGBClassifier` |

## Contenido

| Archivo | Descripción |
|---|---|
| [`Presentacion_Conformal_Prediction.pdf`](Presentacion_Conformal_Prediction.pdf) | Presentación (29 diapositivas) |
| [`Documento_Conformal_Prediction.pdf`](Documento_Conformal_Prediction.pdf) | Documento de investigación, guion y preguntas frecuentes |
| [`conformal_prediction_ejemplo.ipynb`](conformal_prediction_ejemplo.ipynb) | Notebook con el ejemplo numérico completo, ya ejecutado |
| `scripts/figuras_presentacion.py` | Genera las figuras y las cifras (`figuras/resultados.json`) |
| `scripts/construir_presentacion.py` | Arma la presentación en HTML a partir de las cifras |
| `scripts/ejecutar_notebook.py` | Ejecuta el notebook de principio a fin |
| `fuente/` | Fuentes HTML de la presentación y del documento |
| `figuras/` | Figuras usadas en la presentación |

## Cómo reproducir

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
jupyter notebook conformal_prediction_ejemplo.ipynb
```

Luego ejecute todas las celdas (*Run All*). Tarda menos de un minuto.

Para regenerar las figuras y la presentación:

```bash
python scripts/figuras_presentacion.py
python scripts/construir_presentacion.py
```

## Datos

Solo se usan datos públicos:

- **Regresión:** datos sintéticos generados en el notebook (semilla 42).
- **Fraude:** [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) de Kaggle. El notebook lo descarga con `kagglehub` la primera vez. Si Kaggle pide autenticación, descargue `fraudTrain.csv` y `fraudTest.csv` y déjelos en la carpeta `data/`.

## Resultados principales

**Regresión (promesa del 90 %)**

| Método | Cobertura | Ancho promedio |
|---|---|---|
| Split conformal | 91,5 % | 7,56 |
| CQR | 90,4 % | 6,61 |

**Fraude (datos de prueba, 60.000 transacciones)**

| Regla | Recall | Precisión | Alertas |
|---|---|---|---|
| Umbral 0,5 | 87,0 % | 75,1 % | 402 |
| Split conformal 95 % | 62,2 % | 99,5 % | 217 |
| Mondrian 95 % | 96,0 % | 35,3 % | 943 |

En los meses siguientes (555.719 transacciones) el recall de Mondrian baja a 92,7 %: la promesa solo se mantiene mientras los datos nuevos se parezcan a los de calibración.

## Referencias

- Vovk, Gammerman y Shafer (2005). *Algorithmic Learning in a Random World*.
- Angelopoulos y Bates (2023). *Conformal Prediction: A Gentle Introduction*.
- Romano, Patterson y Candès (2019). *Conformalized Quantile Regression*.
- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE): Model Agnostic Prediction Interval Estimator.
