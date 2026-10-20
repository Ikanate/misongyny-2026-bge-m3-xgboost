# MiSongyny 2026 @ IberLEF 2026

Solución para la detección de misoginia en letras de canciones en español.

**Resultados:** 2do lugar en Task 1 y Task 3.

Para confirmar los resultados, puedes revisar el paper del equipo IReL_IIT(BHU) [paper](https://ceur-ws.org/Vol-4273/MiSonGyny2026_paper8.pdf).

| Tarea | F1-Score | Rank |
| :--- | :---: | :---: |
| Task 1 | 0.8712 | 2 |
| Task 2 | 0.6813 | 5 |
| Task 3 | 0.8109 | 2 |

> Nota: no se redactó el working-notes paper de este envío. Este repositorio
> se reconstruyó a posteriori.

## Enfoque
- Glosas de letras generadas con un LLM (Gemma4) como features auxiliares [G4]
- Entrenar un Backbone BGE-M3 diferente para cada task [BG3]
- XGBoost sobre embeddings congelados, con Optuna [XG]


## Estructura
...

## Experimentos ablativos

**Hiperparámetros:**
- `BATCH_SIZE`: 4
- `EPOCHS`: 10
- `LR_BACKBONE`: 5e-6
- `LR_HEADS`: 1e-3
- `WEIGHT_DECAY`: 0.1
- `WARMUP_RATIO`: 0.1
- `SEED`: 30



### Validación local (Task 1 y 3, split 70/30 en datos de entrenamiento)
#### Configuración Experimental

El pipeline experimental se estructuró en dos fases:
1. **Entrenamiento de Backbones:** Se ajustaron modelos base BGE-m3 (**BG3**) de forma independiente para cada tarea (**Task 1** y **Task 3**), tanto con enriquecimiento de glosas (**G4**) como sin ellas. Esto originó **4 backbones especializados**:
   - `BG3 (T1)`: Task 1 sin glosas
   - `BG3 + G4 (T1)`: Task 1 con glosas
   - `BG3 (T3)`: Task 3 sin glosas
   - `BG3 + G4 (T3)`: Task 3 con glosas
2. **Clasificación con XGBoost:** Posteriormente, se extrajeron las representaciones vectoriales de estos 4 backbones y se entrenó un clasificador **XGBoost** sobre cada uno de ellos para evaluar si un ensamble tabular mejoraba las predicciones directas.

---

#### Resultados Comparativos

| Tarea Objetivo | Backbone Base | Glosas (G4) | Clasificador | Val F1 |
|:---|:---|:---:|:---|:---:|
| **Task 1** | BG3 (T1) | ❌ | Head Lineal (Directo) | 0.8109 |
| | BG3 (T1) | ❌ | XGBoost | 0.8123 |
| | BG3 + G4 (T1) | ✅ | Head Lineal (Directo) | **0.8473** |
| | BG3 + G4 (T1) | ✅ | XGBoost | 0.8328 |
|:---|:---|:---:|:---|:---:|
| **Task 3** | BG3 (T3) | ❌ | Head Lineal (Directo) | 0.7904 |
| | BG3 (T3) | ❌ | XGBoost | 0.7916 |
| | BG3 + G4 (T3) | ✅ | Head Lineal (Directo) | 0.7993 |
| | BG3 + G4 (T3) | ✅ | XGBoost | **0.8325** |

### Tarea 1 (`f1_t1`)

* **Mejor resultado:** **BG3 + G4** (**0.847290**).
* **Impacto de componentes:**
  * **Inclusión de glosas (G4):** La adición de `G4` genera una mejora sistemática en el desempeño:
    * `BG3` incrementa de **0.810858** a **0.847290** (+3.64%).
    * `BG3 + XGBOOST` incrementa de **0.812298** a **0.832797** (+2.05%).
  * **Uso de XGBOOST:** Añadir XGBOOST sobre la base simple (`BG3`) ofrece una mejora mínima (+0.14%), mientras que al combinarse con glosas (`BG3 + G4`), el clasificador directo supera a la variante con XGBOOST por casi 1.5 puntos.

---

### Tarea 3 (`f1_t3`)

* **Mejor resultado:** **BG3 + G4 + XGBOOST** (**0.832520**).
* **Impacto de componentes:**
  * **Sinergia completa:** A diferencia de T1, en T3 el rendimiento óptimo requiere la integración conjunta de **glosas + XGBOOST**, superando con claridad al resto de variantes.
  * **Inclusión de glosas (G4):**
    * Sin XGBOOST, la ganancia aportada por `G4` es reducida (**0.799309** vs **0.790445**).
    * En conjunto con XGBOOST, el incremento es sustancial: pasa de **0.791599** a **0.832520** (+4.09%).
  * **Línea base sin glosas:** Las variantes sin enriquecimiento de glosas (`BG3` y `BG3 + XGBOOST`) se estancan en torno a **0.79**, representando el rendimiento base inferior para esta tarea.


  
## Cómo reproducir
...

## 
