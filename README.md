# bank-marketing-ml_angie_avenda-o
Predicción de suscripción a depósito a plazo (Bank Marketing)

# Predicción de suscripción a depósito a plazo (Bank Marketing)

Proyecto final de Data Science II. Modelo de clasificación binaria que estima la
probabilidad de que un cliente contrate un depósito a plazo tras una campaña telefónica,
para priorizar a quién llamar con un presupuesto de contactos limitado.

## Datos
- **Fuente:** Bank Marketing, UCI Machine Learning Repository (id 222).
- **Tamaño:** 45.211 clientes, 16 variables y el objetivo `y` (11,7% de "yes").
- El notebook descarga los datos automáticamente con `ucimlrepo`; no hace falta
  subir ningún archivo.

## Decisiones clave
- **Se excluye `duration`:** solo se conoce después de la llamada, así que usarla sería
  data leakage.
- **Nulos tratados como categoría** en `poutcome`, `contact`, `education` y `job`,
  tras comprobar que `poutcome` nulo equivale a "sin campaña previa" (99,99%).
- **Variables creadas:** `balance_log`, `campaign_cap`, `previous_cap`,
  `contactado_antes` y `pdays_clean`.
- **Sin leakage:** el preprocesamiento vive dentro de un `Pipeline` y se ajusta solo con
  los datos de entrenamiento en cada fold.

## Resultados
Validación cruzada de 5 folds sobre entrenamiento:

| Modelo | ROC AUC | PR AUC |
|---|---|---|
| Regresión Logística | 0,766 | 0,406 |
| Random Forest | 0,793 | 0,446 |
| XGBoost | 0,800 | 0,452 |

Modelo final: XGBoost optimizado con `RandomizedSearchCV`. En el conjunto de test
(usado una sola vez): **ROC AUC = 0,809 y PR AUC = 0,471** (piso por azar: 0,117).
Contactando al 10% de clientes mejor rankeados se alcanza al 45,0% de los compradores.

Explicabilidad con SHAP: resumen global, ranking por barras y dos casos individuales.

## Limitaciones
- Los datos vienen de una campaña pasada con selección no aleatoria de clientes.
- `contact_desconocido` se concentra en mayo y junio; no debe leerse como efecto del canal.
- SHAP describe cómo usa el modelo las variables, no relaciones causales.

## Cómo reproducirlo
1. Abrí `bank_marketing.ipynb` en Google Colab (o en local con Python 3.13).
2. Instalá las dependencias: `pip install -r requirements.txt`
3. Ejecutá todas las celdas en orden. Tarda unos 15 a 25 minutos por la búsqueda de
   hiperparámetros. Todas las semillas están fijadas (`random_state=42`).
