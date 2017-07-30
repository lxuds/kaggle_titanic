# kaggle_titanic

In this competition, we were asked to predict what sorts of people would survive in the Titanic tragedy. 

Data: 1502 out of 2224 passengers and crew were killed after the Titanic sank. The training sets are labeled, and include the informatin of 891 passengers, such as Sex, Age, Fare, Ticket class, Number of siblings/spouses aboard the Titanic, etc. The testing sets contain the same information of 418 passengers. You may find the data and more description at https://www.kaggle.com/c/titanic

Since the dataset is small and algorithms are relatively simple, we used iPython notebook to perform data visualization and feature engineering for this project. 

First step: Data visualization and feature engineering. 

Second step:  We used a stacker classifier (second-level model) to predict the output from the earlier first-level predictions. For the first-level predictions, we chose four base models: Extra Trees, Gradient Boosted Trees, Random Forest, and Xgboot. We used grid search to find the best set of parameters for each model. Then we performed the 5-fold stacking to build the ensemble of classifiers and make the second-level predictions. The stacker classifier we used here is XGBClassifier.

Code Description:

- preprocess.ipynb: This file performs data visualization and feature engineering.
- ensemble.py: This file trains the stacker classifier.
- In folder model:
  - extratrees.py: This files finds the best parameters for Extra Trees model, and generate the first-level predictions.
  - gbm.py: This files finds the best parameters for Gradient Boosted Trees, and generate the first-level predictions.
  - randomforest.py: This files finds the best parameters for Random Forest, and generate the first-level predictions.
  - xgb.py: This files finds the best parameters for Xgboot, and generate the first-level predictions.
  






