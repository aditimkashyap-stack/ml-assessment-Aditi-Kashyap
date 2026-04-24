B1. Problem Formulation 

(a) Target, Features, Problem Type 

The target variable would be the number of items sold per store per month.

Candidate input features:

Promotion type (categorical: Flat Discount, BOGO, etc.)

Store attributes (size, location type, footfall, competition density, demographics)

Calendar features (month, season, weekend/festival flags)

Historical sales trends (lagged sales volume, moving averages).

Problem type: Supervised learning, specifically multi-class classification or regression.

If framed as regression: predict sales volume given promotion + context.

If framed as a classification problem, predict which promotion maximises items sold.

Justification: Since the goal is to recommend the best promotion, regression (predict sales volume) followed by choosing the promotion with the highest predicted value would be most appropriate.
