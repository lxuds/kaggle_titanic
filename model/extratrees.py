import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import pipeline, grid_search

random.seed(2016)



def main():

    start_time = time.time() 
    input1='../data/preprocessing_train_df.csv'
    input2='../data/preprocessing_test_df.csv'
    
    #load preprocessed training data as a dataframe
    train_df = pd.read_csv(input1, index_col=0)
    test_df  = pd.read_csv(input2, index_col=0)

    x_train = train_df.iloc[:, 2:]
    y_train = train_df.iloc[:, 1]

    id_test = test_df['PassengerId']
    x_test = test_df.iloc[:,1:]
    
    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(x_train.columns.tolist()))
    

# Grid Search for ExtraTreesClassifier    
    
    exc = ExtraTreesClassifier(random_state=2016, n_jobs=1, verbose=1)
    param_grid = {"n_estimators": [50], #[50, 100, 400, 700, 1000],
                 "max_features":[20] #[7,8,10,14,18,20,25,30]
                 }

    model = grid_search.GridSearchCV(estimator=exc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=5, verbose=20)
    model.fit(x_train, y_train)
    
    
    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Param grid score:')
    print(model.grid_scores_)    
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(model.best_score_)

    
    y_pred = model.predict(x_test)
    pd.DataFrame({'PassengerId': id_test, 'Survived': y_pred}).to_csv('../data/submission_ext.csv', index=False)
    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


if __name__ == '__main__':
    main()
