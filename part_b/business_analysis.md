B1. Problem Formulation
(a) Target, Features, Problem Type
	Target variable: Number of items sold per store per month.
	Candidate input features:
	Promotion type (categorical: Flat Discount, BOGO, etc.)
	Store attributes (size, location type, footfall, competition density, demographics)
	Calendar features (month, season, weekend/festival flags)
	Historical sales trends (lagged sales volume, moving averages).
	Problem type: Supervised learning, specifically multi-class classification or regression.
	If framed as regression: predict sales volume given promotion + context.
  If framed as a classification problem, predict which promotion maximises items sold.
  Justification: Since the goal is to recommend the best promotion, regression (predict sales volume) followed by choosing the promotion with the highest    predicted value is most appropriate.

(b) Items Sold vs Revenue 
	Why items sold better:
	Revenue is influenced by price variation (discount depth, product mix).
	Promotions like BOGO or gifts may reduce revenue per item but increase units sold, which is the stated business goal.
	Broader principle: The target variable must align with the true business objective. In ML projects, misaligned targets lead models to optimise the wrong  outcome.

(c) Modelling Strategy 
	Issue with global model: Different store contexts (urban vs rural, high vs low competition) mean promotions have heterogeneous effects.
	Alternative:
	Hierarchical/multi-level modelling (global model with store-level random effects).
	Or cluster stores by similarity (footfall, demographics) and train separate models per cluster.
	Justification: Captures local variation while leveraging shared patterns across stores.


B2. Data and EDA Strategy 
(a) Data Integration
	Join strategy:
	Transactions → aggregated by store-month.
	Join with store attributes (store_id key).
	Join with promotion details (promotion_id, store_id, month).
	Join with calendar (month, festival/weekend flags).
	Grain of final dataset: One row = store-month-promotion combination.
	Aggregations:
	Total items sold, total revenue, average basket size.
	Promotion participation rate (% transactions under promotion).
	Footfall aggregated monthly.

(b) EDA Analyses 
1.	Promotion effectiveness plots: Bar charts of average items sold by promotion type. → Guides feature importance and baseline comparisons.
2.	Store segmentation: Boxplots of sales volume by location type (urban/semi-urban/rural). → Suggests clustering or stratification.
3.	Seasonality trends: Line charts of monthly sales volume across years. → Motivates inclusion of calendar features.
4.	Correlation heatmap: Between store attributes (size, footfall, competition) and sales volume. → Identifies multicollinearity, informs feature selection.

(c) Imbalance in Promotions 
	Effect: The model may learn “no promotion” as the default and underfit promotion effects.
	Steps:
	Resampling (oversample promotion cases, undersample no-promotion).
	Weighting loss function to emphasise promotion rows.
	Ensure balanced evaluation metrics (precision/recall per promotion class).


B3. Model Evaluation and Deployment 
(a) Train-Test Split & Metrics
	Split strategy:
	Time-based split: train on the first 2 years, validate/test on the last year.
	Random split would be inappropriate since it leaks future information into training.
	Metrics:
	RMSE/MAE (for regression) → measures prediction error in items sold.
	Uplift in items sold vs baseline promotion policy → business impact.
	Precision/recall per promotion (if classification) → ensures minority promotions are evaluated fairly.

(b) Feature Importance Communication 
	Investigation:
	Use SHAP values or permutation importance to see which features drove December vs March predictions.
	Example: December → festival flag + loyalty program engagement high → Loyalty Points Bonus recommended.
	March → high competition + low footfall → Flat Discount recommended.
	Communication: Present simple visuals (bar charts of top features per month) and explain in business terms: “The model chose Loyalty Points Bonus in December because customer loyalty and festival shopping were strong drivers.”

(c) Deployment Process 
	Steps:
1.	Save model: Serialize with joblib/pickle (Python) or MLflow for versioning.
2.	Data pipeline: Each month, aggregate new transaction + store + promotion + calendar data into the same schema.
3.	Prediction service: Feed monthly store-level features into the model, generate promotion recommendations.
4.	Monitoring:
	Track prediction accuracy vs actual items sold.
	Monitor drift in input features (e.g., footfall distribution changes).
	Retrain when performance drops below the threshold or new promotion types are introduced.

