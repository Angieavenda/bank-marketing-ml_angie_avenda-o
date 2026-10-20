# Predicción de suscripción a depósito a plazo (Bank Marketing)

Proyecto final de Data Science II. Modelo de clasificación binaria que estima la probabilidad de que un cliente contrate un depósito a plazo tras una campaña telefónica, para priorizar a quién llamar con un presupuesto de contactos limitado.

## Contenido del repositorio
- `bank_marketing.ipynb`: notebook completo (EDA, ingeniería de atributos, modelos, optimización, evaluación y SHAP), con todas las salidas.
- `requirements.txt`: librerías necesarias.
- `README.md`: este archivo.

## Datos
- **Fuente:** Bank Marketing, UCI Machine Learning Repository (id 222).
- **Tamaño:** 45.211 clientes y 16 variables originales, más el objetivo `y` (11,7% de "yes").
- El notebook descarga los datos automáticamente con `ucimlrepo`; no hace falta subir ningún archivo.

## Variables excluidas o transformadas
El modelo final usa 16 variables, resultado de las decisiones de esta tabla:

| Variable original | Tratamiento | Motivo |
|---|---|---|
| `duration` | Excluida | Solo se conoce después de la llamada, así que usarla sería data leakage. Un `assert` detiene el notebook si reaparece en los datos de entrenamiento. |
| `balance` | Reemplazada por `balance_log` | Fuerte asimetría y valores extremos; se aplica logaritmo con signo. |
| `campaign` | Reemplazada por `campaign_cap` | Cola larga (máximo 63); se acota en 10. |
| `previous` | Reemplazada por `previous_cap` | Cola larga; se acota en 5. |
| `pdays` | Reemplazada por `pdays_clean` y `contactado_antes` | El valor -1 es un código de "sin contacto previo", no una cantidad de días. |
| `day_of_week` | Renombrada a `day_of_month` | Contiene valores de 1 a 31: es el día del mes. |

## Decisiones clave
- **Nulos tratados como categoría** en `poutcome`, `contact`, `education` y `job`. En `poutcome`, el nulo equivale a "sin campaña previa" (99,99% de los casos).
- **Sin leakage:** el preprocesamiento vive dentro de un `Pipeline` y se ajusta solo con los datos de entrenamiento en cada fold.
- **Métricas:** ROC AUC y PR AUC, porque accuracy engaña con un desbalance de ~88/12.

## Resultados
Validación cruzada de 5 folds sobre entrenamiento:

| Modelo | ROC AUC | PR AUC |
|---|---|---|
| Regresión Logística | 0,766 | 0,406 |
| Random Forest | 0,793 | 0,446 |
| XGBoost | 0,800 | 0,452 |

Modelo final: XGBoost optimizado con `RandomizedSearchCV` (20 combinaciones × 5 folds). La mejora sobre la configuración inicial fue marginal (PR AUC en CV de 0,454 frente a 0,452).

En el conjunto de test (usado una sola vez): **ROC AUC = 0,809 y PR AUC = 0,471** (piso por azar: 0,117). Contactando al 10% de clientes mejor rankeados se alcanza al 45,0% de los compradores.

Explicabilidad con SHAP: resumen global, ranking por barras y dos casos individuales.

## Limitaciones
- Los datos vienen de una campaña pasada con selección no aleatoria de clientes.
- `contact_desconocido` se concentra en mayo y junio; no debe leerse como efecto del canal.
- SHAP describe cómo usa el modelo las variables, no relaciones causales.

## Cómo reproducirlo
1. Abrí `bank_marketing.ipynb` en Google Colab (o en local con Python 3.13).
2. Instalá las dependencias: `pip install -r requirements.txt`
3. Ejecutá todas las celdas en orden. Tarda unos 15 a 25 minutos por la búsqueda de hiperparámetros. Todas las semillas están fijadas (`random_state=42`).

Los resultados de este README se obtuvieron con Python 3.13.16 y las versiones que muestra la última celda del notebook. Con otras versiones de las librerías (en especial `xgboost`, fijado como `>=2.0`) las métricas pueden variar en el tercer decimal.
