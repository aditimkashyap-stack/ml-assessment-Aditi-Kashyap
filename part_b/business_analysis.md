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

(b) Items Sold vs Revenue

Why are the items sold better:

Revenue is influenced by price variation (discount depth, product mix).

Promotions like BOGO or the gifts may reduce revenue per item but increase units sold, which is the stated business goal.

Broader principle: The target variable must align with the true business objective. In ML projects, misaligned targets lead models to optimise the wrong outcome.

(c) Modelling Strategy

Issue with global model: Different store contexts (urban vs rural, high vs low competition) means that promotions have heterogeneous effects.

Alternative method:

Hierarchical/multi-level modelling (global model with store-level random effects).

Or cluster stores by similarity (footfall, demographics) and train separate models per cluster.

Justification: Captures local variation while leveraging shared patterns across stores.
