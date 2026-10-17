# MiSongyny 2026 @ IberLEF 2026

Solución para la detección de misoginia en letras de canciones en español.

**Resultados:** 2do lugar en Task 1 y Task 3.

Para confirmar y consultar los detalles de los resultados, puedes revisar el paper del equipo IReL_IIT(BHU) [paper](https://ceur-ws.org/Vol-4273/MiSonGyny2026_paper8.pdf).

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

#### Validación local (Task 1, split 70/30 en datos de entrenamiento)
| Modelo / Enfoque | F1-Score |
| :--- | :---: |
| BG3 | 0.8172 |
| BG3 + G4 | 0.8389 |

## Cómo reproducir
...

## 
