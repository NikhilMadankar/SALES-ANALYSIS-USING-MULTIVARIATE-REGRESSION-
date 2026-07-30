# BigMart Sales Analysis

Predicting product sales across BigMart retail outlets using regression analysis on the classic BigMart Sales dataset.

##  Objective

BigMart collects daily sales data for various products across 10 stores in different cities. The goal of this project is to:
- Identify which product and outlet attributes most influence sales
- Build a regression model to predict `Item_Outlet_Sales` for products at each outlet
- Understand which features can be dropped without meaningfully hurting model performance

##  Dataset

The dataset is the well-known **BigMart Sales** dataset (originally from an Analytics Vidhya hackathon).

| File | Rows | Description |
|---|---|---|
| `Train.csv` | 8,523 | Labeled data, includes `Item_Outlet_Sales` (target) |
| `Test.csv` | 5,681 | Unlabeled data, target withheld |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `Item_Identifier` | Categorical | Unique product ID (1,559 distinct items) |
| `Item_Weight` | Numeric | Weight of the product (~17% missing) |
| `Item_Fat_Content` | Categorical | Low Fat / Regular (inconsistently labeled) |
| `Item_Visibility` | Numeric | % of display area allotted to the item |
| `Item_Type` | Categorical | 16 product categories |
| `Item_MRP` | Numeric | Maximum Retail Price |
| `Outlet_Identifier` | Categorical | Unique store ID (10 stores) |
| `Outlet_Establishment_Year` | Numeric | Year the store was established |
| `Outlet_Size` | Categorical | Small / Medium / High (missing values present) |
| `Outlet_Location_Type` | Categorical | Tier 1 / Tier 2 / Tier 3 city |
| `Outlet_Type` | Categorical | Grocery Store / Supermarket Type 1, 2, 3 |
| `Item_Outlet_Sales` | Numeric (Target) | Sales of the product at that store |


##  Data Cleaning

- **Missing `Item_Weight`** → filled using the mean weight of the same `Item_Identifier` across other stores; any item with no known weight anywhere filled with the overall column mean.
- **Missing `Outlet_Size`** → filled using the mode `Outlet_Size` within each `Outlet_Type` group.
- **`Item_Visibility` = 0** → replaced with the mean visibility computed from non-zero rows only (a visibility of 0 isn't physically meaningful for a stocked product).
- **Inconsistent `Item_Fat_Content` labels** (`LF`, `reg`, `low fat`, `Low Fat`, `Regular`) → standardized into two clean categories.

##  Feature Engineering

- `New_Item_Type` — derived from the first two letters of `Item_Identifier` (`FD`/`NC`/`DR`) → mapped to Food / Non-consumable / Drinks
- `Age_of_Outlet` — computed as `2013 - Outlet_Establishment_Year`
- One-hot encoding (`pd.get_dummies`) for all categorical variables, with one category dropped per variable to avoid the dummy variable trap
- Label encoding used as an alternate representation for exploratory correlation analysis

## Modeling Approach

- **scikit-learn `LinearRegression`** — fitting, prediction, cross-validated MSE/R²
- **statsmodels `OLS`** — detailed coefficient significance (p-values, confidence intervals)
- **Log-transform** applied to the target variable to correct heteroscedasticity in residuals
- **Variance Inflation Factor (VIF)** used to detect and remove multicollinear features
- **Manual feature subset testing** (1, 2, 3, 4, 6-variable combinations) to find the smallest set preserving most predictive power
- **`SelectKBest` (k=4)** used as an algorithmic cross-check against the manually selected features

##  Results

| Model | R² | CV MSE | Features |
|---|---|---|---|
| Full feature set (log-transformed target) | 0.7206 | 0.2901 | 16 |
| After VIF-based feature removal | 0.7200 | 0.2908 | 15 |
| SelectKBest (k=4) | 0.6943 | 0.3166 | 4 |
| Manually selected 4-feature model | ≈0.72 | — | 4 |
| Without log-transform (raw sales) | 0.5620 | 1,277,054 | 4 |


##  Conclusion

Just 4 features — `Item_MRP` and 3 `Outlet_Type` dummy variables — explain nearly as much variance in sales as the full 16-feature model. Log-transforming the target had a bigger impact on model quality (R² 0.72 vs 0.56) than adding extra features, confirming that correcting for heteroscedasticity mattered more than feature count.

