# World Cup 2026 Predictor

<p align="center">
  <img src="static/fifa.jpg" alt="FIFA" width="350"/>
</p>

**WC26 Predictor** is a personal Machine Learning project focused on predicting international matches, with a special emphasis on the *FIFA World Cup 2026*. 

Using the [**international_results**](https://github.com/martj42/international_results) dataset, the system estimates each national team's offensive and defensive strength through a Dixon-Coles model. With these estimates, two expected-goals prediction approaches are trained:

- A Bayesian MCMC model
- An XGBoost regression model

The goal predictions are transformed into a probability matrix based on `Poisson` distributions, which allows each model to calculate the final 1X2 outcome probabilities.

## Running the project

Create and activate a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Install the dependencies and run the interactive interface:

```bash
pip install -r requirements.txt
python main.py
```

## Visualizing results

Optionally, you can generate visualizations of the results with `pyplot` from the interactive interface.

The generated plots are stored in the `outputs/` directory.

## No-color mode

If your terminal doesn't support ANSI colors, you can use the following command:

```bash
NO_COLOR=1 python main.py
```

> [!IMPORTANT]
> **This project was developed for educational and research purposes.** Although the model may produce interesting predictions, it is not recommended for use in betting, since no model can predict football's inherent uncertainty.
> 

<p align="center">
  <img src="static/apuesta.jpg" alt="Bet" width="500"/>
</p>

That said, the program has personally produced accurate results on a few occasions. This should be considered anecdotal and not a reliable statistical basis.

## Testing

The project has an automated *testsuite* that you can run with:

```bash
pytest -v
<<<<<<< HEAD
```
=======
```

>>>>>>> 0fcc99ab7f3e7ef9d417ada1540b36bcf4a29e9a
