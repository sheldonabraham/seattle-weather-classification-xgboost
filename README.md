# Seattle Weather Prediction

Classifying each day's weather in Seattle (drizzle, fog, rain, snow or sun) from four measurements, using XGBoost.

**Main finding:** the model is about 81% accurate. In practice it works as a wet-day or dry-day detector. It finds rain very well. It can't tell drizzle or fog apart from sun, because those days all have zero precipitation.

## Results (test set: 293 days)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Always guess "sun" (most common type in the test set) | 0.427 | n/a |
| XGBoost, default settings | 0.758 | 0.47 |
| XGBoost, tuned (gamma 1, learning rate 0.1) | **0.809** | 0.46 |

Macro F1 averages the score across all five weather types, so the rare types count as much as rain and sun.

### Tuned model, by weather type

| Weather | Test days | Precision | Recall |
|---|---|---|---|
| Rain | 123 | 0.97 | 0.90 |
| Sun | 125 | 0.72 | 0.98 |
| Snow | 6 | 1.00 | 0.33 |
| Fog | 29 | 0.33 | 0.03 |
| Drizzle | 10 | 0.00 | 0.00 |

Tuning raised accuracy but did not improve macro F1. The tuned model predicts "sun" for almost every dry day, which helps overall accuracy but misses nearly every fog and drizzle day.

## Data

- `seattle-weather.csv`: 1,461 days of Seattle weather, 1 Jan 2012 to 31 Dec 2015, with no missing values
- Inputs: precipitation, maximum temperature, minimum temperature, wind
- Target: weather type. Rain (641 days) and sun (640) make up 88% of the data. Fog (101), drizzle (53) and snow (26) are rare.

## Method

1. Encode the weather types as numbers (drizzle 0, fog 1, rain 2, snow 3, sun 4).
2. Scale each input to 0–1 by dividing by its maximum. Drop the date column.
3. Split 80/20 into training and test sets.
4. Train a default XGBoost classifier as a baseline.
5. Tune learning rate and gamma with a grid search (16 combinations, 10-fold cross-validation, scored by accuracy). Best cross-validated accuracy: 0.868.

## What I learned

- Accuracy can look good on imbalanced data while the model ignores the rare classes. Per-class precision and recall show what is really happening.
- Tuning for accuracy pushes the model towards the common classes.
- A model can only separate classes if the inputs contain the difference. Drizzle, fog and sun days all have zero precipitation, so four measurements aren't enough to tell them apart.

## Limitations and next steps

- Adding humidity, cloud cover or visibility would likely help identify fog and drizzle.
- The test set has only 6 snow and 10 drizzle days, so the scores for these types are unreliable.
- The split is not stratified, so the rare types may be unevenly spread between training and test sets.
- The scaling uses the maximum from the full dataset, before the split. Fitting it on the training data only would be more correct.
- Four years of data from one city.

## Repository contents

```
├── Weather_Prediction.ipynb   # the full analysis
├── seattle-weather.csv        # daily weather data
└── README.md
```

## How to run

```bash
pip install pandas xgboost scikit-learn jupyter
jupyter notebook Weather_Prediction.ipynb
```

Keep `seattle-weather.csv` in the same folder as the notebook, then run all cells.

## Acknowledgements

Built while following a tutorial: [add link here].

## Tools

Python, pandas, XGBoost, scikit-learn
