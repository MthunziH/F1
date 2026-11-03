# 🏁 F1 Podium Predictor

Predicts the **top 3 finishers of every Formula 1 Grand Prix** right after qualifying,
with each driver's chance to win and to finish on the podium. It runs by itself on
GitHub and publishes a phone-friendly page:

**➡️ https://mthunzih.github.io/F1/**

## What happens automatically

| When | What |
|---|---|
| Friday–Sunday, hourly | Checks if qualifying has finished. If so, it predicts the race, updates the page, and opens a GitHub issue that notifies you. |
| After each race | Records the real result, scores the prediction (winner right? how many of the podium?), comments on and closes the issue, and retrains with the new race. |

Everything is stored in the repo: `data/history.csv` (all results since 2018),
`predictions/*.json` (one file per race) and `data/backtest.json`.

## Running it by hand (from your phone)

GitHub app or website → **Actions** → **F1 prediction** → **Run workflow**:

- `auto`: do whatever is due right now
- `predict`: a specific race. Fill in season + round, and tick *force* to redo it.
- `backtest`: re-measure accuracy on past races

## How the prediction works

1. **Data** comes from [FastF1](https://github.com/theOehrly/Fast-F1). Past results come
   through its Ergast/Jolpica interface. The current weekend's qualifying comes from F1 live timing,
   with Ergast as a fallback.
2. **Features** (only what's known before the race starts): qualifying position, gap to
   pole, team's best qualifying, gap to teammate, driver form over the last 5/10 races (avg finish,
   points, podium/win/retirement rate, places gained), season points share, team form, the driver's
   record at this circuit, and how often pole converts to a win at this circuit.
3. **Model**: two gradient-boosting classifiers (scikit-learn), one for *win* and one for *podium*.
   They're retrained before every race on all earlier races. Probabilities are normalised per
   race (win chances add up to 100%, podium chances to 300%).
4. **Predicted podium**: P1 is the most likely winner. P2 and P3 are the next most likely podium finishers.
5. **Backtest**: every race since 2024 is replayed using only data from before it, and
   compared with the simple "qualifying top 3 = podium" rule.

Limitations: grid penalties announced after qualifying, rain, crashes and safety cars
aren't known in advance.

## Local use

```bash
pip install -r requirements.txt
python -m f1predict auto                          # same as the scheduled job
python -m f1predict predict --season 2026 --round 20
python -m f1predict backtest
pytest -q                                         # offline tests (synthetic data)
```
