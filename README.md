# housing-price-prediction

Predicting median home values in California from property and location features, comparing a linear regression baseline against a Random Forest model.
 
## Data
 
- **Source:** [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (built into scikit-learn, derived from the 1990 U.S. Census)
- **Size:** 20,640 records
- **Features:** median income, house age, average rooms/bedrooms, population, average occupancy, latitude, longitude
- **Target:** median house value (in $100,000s)
## Method
 
1. Split the data 80/20 into training and test sets, so the model is always evaluated on data it never saw during training
2. Standardized all features (mean 0, standard deviation 1) so coefficients and importances are directly comparable across features with very different natural scales
3. Fit two models for comparison:
   - **Linear Regression** — a single straight-line relationship across all features
   - **Random Forest** (200 trees) — averages many decision trees to capture non-linear patterns and feature interactions
4. Evaluated both on the held-out test set using RMSE and R²
## Results
 
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.746 (~$74,600 error) | 0.576 (58% of variance explained) |
| **Random Forest** | **0.504 (~$50,400 error)** | **0.806 (81% of variance explained)** |
 
![Predicted vs actual home values](predicted_vs_actual.png)
 
*Linear regression's predictions vs. actual values. Points near the red dashed line are accurate; the model consistently under-predicts the most expensive homes.*
 
## Findings
 
- **Random Forest substantially outperformed linear regression** — about a third less error and 23 percentage points more variance explained — because it can capture non-linear relationships and feature interactions that a straight-line model can't
- The two models disagree on what matters most, and that disagreement is informative rather than contradictory:
  - Linear regression's coefficients point to **location** (latitude/longitude) as the strongest standalone predictor
  - Random Forest's feature importances point to **median income** as dominant (53%), with location playing a smaller supporting role
  - This suggests income's effect on price is highly interactive with other features — something only the non-linear model can represent
- Linear regression consistently under-predicts the highest-value homes, and produces a few nonsensical negative price predictions — both expected limitations of a purely linear approach
## Tools
 
Python · pandas · scikit-learn (LinearRegression, RandomForestRegressor, StandardScaler, train_test_split) · matplotlib
 
## Notes / limitations
 
- The dataset caps recorded home values at $500,001 (a known quirk of the source Census data) — any home actually worth more is recorded at that ceiling, which visibly distorts predictions near that edge in the scatter plot
- Data reflects 1990 census figures; useful for demonstrating methodology, not current market values
- Neither model was hyperparameter-tuned; a natural next step would be tuning Random Forest's tree depth and count, though the larger accuracy gains would more likely come from adding richer features (e.g. distance to coast, school ratings) than from tuning alone
