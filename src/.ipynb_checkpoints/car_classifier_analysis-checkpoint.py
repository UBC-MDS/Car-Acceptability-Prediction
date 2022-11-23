# date: 2022-11-20
'''This script takes the file path of traning data, then does machine learning analysis 
(muti classififer tuning)(hyperparameter tuning) (final model fitting),
then save the tables/model to the result folder.

Usage: src/eda_car_popularity.py --training_data_path=<training_data_path> --result_folder_path=<result_folder_path>

Options: 
--training_data_path=<training_data_path> The path the the csv file contianing training data
--result_folder_path=<result_folder_path> The path to store the analysis results
'''


from docopt import docopt 
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import RandomizedSearchCV
opt = docopt(__doc__)

def main(training_data_path, result_folder_path):
    try:
        # read data and add column names 
        train_df = pd.read_csv(training_data_path)
    except Exception as req:
        print("The training data file path does not exist, check again")
        print(req)
    
    # pre_processing -----------------------------------------------------------
    # list features for different transformations
    ordinal_features = ['buying', 'maint', 'persons', 'lug_boot', 'safety']
    drop_features = ['doors']
    target = "class"

    # set ordinal levels 
    buying_maint_level = ['low', 'med', 'high', 'vhigh']
    person_level = ['2', '4', 'more']
    lug_level = ['small', 'med', 'big']
    safety_level = ['low' , 'med', 'high']

    # separate target and input features
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    # separate target and input features
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    # create processor
    ordinal_transformer = OrdinalEncoder(categories=[buying_maint_level,buying_maint_level,
        person_level,lug_level,safety_level], dtype=int)
    preprocessor = make_column_transformer(
    ("drop", drop_features),
    (ordinal_transformer, ordinal_features))

    # this can be used in literate documents
    # data = preprocessor.fit_transform(X_train)
    # transformed_df = pd.DataFrame(data, columns=ordinal_features)

    # classifier tuning -----------------------------------------------------------
    # list different classifiers
    models = {
    "dummy": DummyClassifier(random_state = 522),
    "decision tree": DecisionTreeClassifier(random_state=522),
    "random forest": RandomForestClassifier(random_state=522),
    "naive bayes": MultinomialNB()}
    
    # classifier tuning, we use `balanced_accuracy` as our scoring metric 
    muti_model_results_df = classifier_tunning(models,preprocessor,X_train,y_train)
    # this can be used in literate documents
    # print(muti_model_results_df)
    # save table as csv file 
    write_to_csv(muti_model_results_df, result_folder_path+'/car_classifier_tuning_analysis.csv')
    # hyperparameter tunning------------------------------------------------------------
    # create pipeline for random forest hyperparameter tunning 
    pipe_rf = Pipeline([
    ("process", preprocessor),
    ("rf", RandomForestClassifier(random_state = 522))])
    # create hyperparameter grid 
    param_dist = {
    "rf__n_estimators": range(10,71,1),
    "rf__max_depth": range(3,20,1),}
    # hyperparameter optimization
    random_search_result = hyperparameter_tuning(pipe_rf, param_dist,X_train,y_train)
    # this can be used in literate documents
    # print(random_search_result.head(5))
    # save table as csv file 
    write_to_csv(random_search_result, result_folder_path+'/car_hyperparameter_tuning_analysis.csv')

    # final model-----------------------------------------------------------------
    best_n_estimators, best_max_depth = 57, 8
    final_model = make_pipeline(preprocessor,  RandomForestClassifier(random_state=522, 
        n_estimators= best_n_estimators, max_depth= best_max_depth))
    final_model.fit(X_train, y_train)
    # save final model as joblib 
    try:
        joblib.dump(final_model, result_folder_path+"/final_model.joblib")
    except:
        os.makedirs(os.path.dirname(result_folder_path))
        joblib.dump(final_model, result_folder_path+"/final_model.joblib")
    







# reference to DSCI 571
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train,scoring = "balanced_accuracy", **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

def classifier_tunning(models,preprocessor,X_train,y_train):
    results_dict = {}
    muti_model_results_df = None 
    for model_name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results_dict[model_name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=5, return_train_score=True
        )
    muti_model_results_df = pd.DataFrame(results_dict).T

    return muti_model_results_df

def hyperparameter_tuning(pipe_rf, param_dist,X_train,y_train):
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations
    random_search = RandomizedSearchCV(pipe_rf, param_dist, cv=5, 
        n_iter=100,n_jobs=-1, verbose=1,scoring = "balanced_accuracy",random_state = 522)
    random_search.fit(X_train, y_train)
    random_search_result = pd.DataFrame(random_search.cv_results_)[
    [
        "mean_test_score",
        "param_rf__n_estimators",
        "param_rf__max_depth",
        "mean_fit_time",
        "rank_test_score",
    ]].set_index("rank_test_score").sort_index()
    return random_search_result



# function for writing csv file to given folder path
def write_to_csv(df, file_path):
    try:
        df.to_csv(file_path, index = False)
    except:
        os.makedirs(os.path.dirname(file_path))
        df.to_csv(file_path, index = False)




if __name__ == "__main__":
    main(opt["--training_data_path"], opt["--result_folder_path"])
