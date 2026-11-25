# BigMart Sales Analysis

Predicting product sales (`Item_Outlet_Sales`) across BigMart retail outlets using a leak-free multivariate linear regression pipeline on the BigMart Sales dataset.

## Objective

BigMart collects product-level sales data across 10 retail outlets of varying sizes, formats, and city tiers. The goal of this project is to:
- Identify which product and outlet attributes most strongly drive `Item_Outlet_Sales`
- Build a statistically valid, leak-free OLS / Linear Regression model
- Address core linear regression assumptions (heteroscedasticity and multicollinearity)
- Find the smallest, most interpretable feature subset that generalizes best on unseen holdout data (Occam's Razor)

## Dataset

The project uses `Train1.csv` from the **BigMart Sales** dataset (**8,523 rows × 12 columns**), split **80/20** (`6,818` training / `1,705` unseen holdout test rows, `random_state=42`) **before any preprocessing** to prevent data leakage.

| Column | Type | Description |
|---|---|---|
| `Item_Identifier` | Categorical | Unique product ID (`1,559` distinct items) |
| `Item_Weight` | Numeric | Weight of the product (`1,463` missing, ~17.2%) |
| `Item_Fat_Content` | Categorical | Low Fat / Regular (dirty labels: `LF`, `low fat`, `reg`) |
| `Item_Visibility` | Numeric | % of display area allotted to the item (`526` zero anomalies) |
| `Item_Type` | Categorical | 16 product categories |
| `Item_MRP` | Numeric | Maximum Retail Price |
| `Outlet_Identifier` | Categorical | Unique store ID (`10` stores) |
| `Outlet_Establishment_Year` | Numeric | Year the store was established (`9` distinct years) |
| `Outlet_Size` | Categorical | Small / Medium / High (`2,410` missing, ~28.3%) |
| `Outlet_Location_Type` | Categorical | Tier 1 / Tier 2 / Tier 3 city |
| `Outlet_Type` | Categorical | Grocery Store / Supermarket Type 1, 2, 3 |
| `Item_Outlet_Sales` | Numeric (Target) | Sales of the product at that store |

## Pipeline (8 Stages)

1. **Data Loading & Exploration + Train–Test Split**: Schema, missingness, cardinality, anomaly checks, and an 80/20 holdout split *before* preprocessing.
2. **Data Cleaning & Imputation (Fit on Train Only)**:
   - Missing `Item_Weight` → imputed with training mean per `Item_Identifier` (fallback to overall training mean).
   - Missing `Outlet_Size` → imputed with training mode per `Outlet_Type`.
   - `Item_Visibility == 0` → replaced with training mean of non-zero visibility values.
   - `Item_Fat_Content` → standardized (`LF`, `low fat` → `Low Fat`; `reg` → `Regular`).
3. **Feature Engineering**:
   - `New_Item_Type` → extracted from first 2 characters of `Item_Identifier` (`FD` → `Food`, `NC` → `Non-consumable`, `DR` → `Drinks`).
   - `Age_of_Outlet` → `2013 - Outlet_Establishment_Year`.
   - Dropped raw IDs and redundant columns (`Item_Identifier`, `Outlet_Identifier`, `Outlet_Establishment_Year`, `Item_Type`).
4. **Encoding & Correlation**:
   - Proper One-Hot Encoding (`pd.get_dummies(..., dtype=float)`) for all nominal categoricals (no `LabelEncoder` on nominal IDs), dropping one explicit reference level per category → **14 clean features**.
   - Test columns aligned via `.reindex(..., fill_value=0.0)`.
5. **Target Transformation**:
   - Natural log transformation (`ln(Item_Outlet_Sales)`) to address right-skewness and heteroscedasticity observed in raw residuals.
6. **Modeling & Diagnostics**:
   - Leak-free **5-fold cross-validation** using a custom scikit-learn `BigMartPreprocessor` inside a `Pipeline`.
   - Full OLS summary via `statsmodels` and **proper iterative VIF pruning** (with intercept constant included via `sm.add_constant`) until all remaining features have `VIF <= 5`.
7. **Feature Selection & Final Choice**:
   - Compared single-regressor baselines, incremental subsets, leak-free `SelectKBest(f_regression, k=4)`, and the complete 4-feature group (`Item_MRP` + 3 `Outlet_Type` dummies; all VIFs ≤ 2.38).
8. **Holdout Test Validation & Residual Diagnostics**:
   - Apples-to-apples comparison on the **original currency scale** using **Duan's smearing estimator** (learned strictly from training residuals) to back-transform log predictions.
   - 4-panel residual diagnostic suite on the final 4-feature model (Residuals vs. Fitted for Raw vs. Log, Normal Q-Q Plot, and Holdout Actual vs. Predicted Sales).

## Holdout Results (`1,705` Unseen Test Rows)

| Model | Features | 5-Fold CV Log RMSE | Test Log R² | Test Orig R² | Test Orig RMSE | Test Orig MAE | Neg. Preds | AIC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model (Log) | 14 | 0.5400 | 0.7296 | 0.5549 | 1,099.83 | 754.54 | 0 | 10,952.9 |
| VIF-Pruned Model (Log) | 12 | 0.5410 | 0.7267 | 0.5491 | 1,107.08 | 760.77 | 0 | 10,976.4 |
| SelectKBest `k=4` (Log) | 4 | 0.5642 | 0.7018 | 0.5151 | 1,148.05 | 800.31 | 0 | 11,548.6 |
| **Final Top-4 Model (Log + Smearing)** | **4** | **0.5399** | **0.7295** | **0.5571** | **1,097.21** | **753.61** | **0** | **10,944.1** |
| Top-4 Model (Raw Target, No Log) | 4 | — | — | 0.5801 | 1,068.28 | 791.22 | 93 | 115,387.7 |

## Conclusion

- **Occam's Razor Confirmed**: Just **4 features** — `Item_MRP` and the 3 `Outlet_Type` dummy variables (`Grocery Store`, `Supermarket Type1`, `Supermarket Type2`, with `Supermarket Type3` as reference) — outperform the full 14-feature model on leak-free CV Log RMSE (`0.5399`), holdout RMSE (`1,097.21`), holdout MAE (`753.61`), and AIC (`10,944.1`), with zero multicollinearity (`VIF <= 2.38`).
- **Why Log Transformation Wins in Practice**: While raw OLS yields a slightly lower original-scale RMSE (`1,068.28`) by directly minimizing raw squared error, it suffers from severe heteroscedasticity, worse MAE (`791.22` vs. `753.61`), and produces **93 impossible negative sales predictions** (down to `-1,341.81`). The log-linear model with Duan's smearing stabilizes error variance, improves MAE, and guarantees strictly positive sales predictions (`0` negatives).
