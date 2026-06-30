# WC26 PREDICTOR

<p align="center">
  <img src="static/fifa.jpg" alt="FIFA" width="350"/>
</p>

**WC26 Predictor** es un proyecto personal de Machine Learning orientado a la predicción de partidos internacionales, con especial enfoque en la *Copa Mundial* de la FIFA 2026. A partir del dataset [**international_results**](https://github.com/martj42/international_results), el sistema estima la fuerza ofensiva y defensiva de cada selección mediante un modelo Dixon-Coles. Con estas estimaciones, se entrenan dos enfoques de predicción de goles esperados:

- Un modelo MCMC Bayesiano
- Un modelo de regresión con XGBoost

Las predicciones de goles se transforman en una matriz de probabilidades basada en distribuciones de `Poisson`, lo que permite calcular las probabilidades finales del resultado 1X2.

## Ejecución

Crea y activa un entorno virtual:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Instala las dependencias y ejecuta la interfaz interactiva:

```bash
pip install -r requirements.txt
python main.py
```

## Visualización de resultados 

Opcionalmente, puedes generar visualizaciones de los resultados con `matplotlib` / `pyplot` desde la interfaz interactiva.
Los gráficos generados se almacenan en el directorio `outputs/`.

## Modo sin colores

Si tu terminal no soporta colores ANSI puedes usar el comando:

```bash
NO_COLOR=1 python main.py
```

> [!IMPORTANT]
> **Este proyecto fue desarrollado con fines educativos y de investigación.** Aunque el modelo puede ofrecer predicciones interesantes, **no se recomienda utilizarlo para realizar apuestas deportivas**, ya que ningún modelo puede predecir un partido con certeza y el fútbol contiene una gran cantidad de factores impredecibles.
>

<p align="center">
  <img src="static/apuesta.jpg" alt="Apuesta" width="500"/>
</p>

Dicho eso, personalmente el programa ha producido resultados acertados en algunas ocasiones. Esto debe considerarse anecdótico y no una base estadística confiable.

## Testing

El proyecto cuenta con una *testsuite* automatizada que puedes ejecutar con:

```bash
pytest -v
```
