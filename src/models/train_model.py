# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, \
                            classification_report, \
                            accuracy_score,\
                            balanced_accuracy_score,\
                            ConfusionMatrixDisplay,\
                            RocCurveDisplay, DetCurveDisplay
#from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
from utils import read_data, remove_missing_values, var_list, numeric_features,\
                    preprocessor, categorical_features, categorical_transformer, numeric_transformer


# Get the current working directory
current_working_directory = os.getcwd()

# Convert the current working directory to a Path object
script_dir = Path(current_working_directory)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_pipeline, model_name):
    """
    This function trains and evaluates model, and generates confusion matrix, classification report, and accuracy score
    Parameters:
    ----------
        X_train
        y_train
        X_test
        y_test
        model_pipeline
        model_name
    Returns:
    -------
        None
    """
    
    model_pipeline.fit(X_train, y_train.values.ravel())


    # Predict
    y_pred = model_pipeline.predict(X_test)
    
    # Obtain accuracy score
    acc = accuracy_score(y_test, y_pred)
    print('accuracy is',accuracy_score(y_pred,y_test))
    
    score_train = model_pipeline.score(X_train, y_train)
    score_test = model_pipeline.score(X_test, y_test)
    print('score for training set', score_train, 'score for testing set', score_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Balanced accuracy score", balanced_accuracy)
    
    report = classification_report(y_test, y_pred)

    
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    _ = ax.set_title(
        f"Confusion Matrix for {model_name}"
    )
    
    return fig, report, score_train, score_test, balanced_accuracy

# +
def classify_grid_search_cv_tuning(model, parameters, X_train, X_test, y_train, y_test, n_folds = 5, scoring='accuracy'):
    """
    This function tunes GridSearchCV model
    
    Parameters:
    ----------
        model
        parameters
        X_train
        X_test
        y_train
        y_test
        n_folds
        scoring
        
    Returns:
    --------
        best_model
        best_score
    """
    # Set up and fit model
    tune_model = GridSearchCV(model, param_grid=parameters, cv=n_folds, scoring=scoring)
    tune_model.fit(X_train, y_train.values.ravel())
    
    best_model = tune_model.best_estimator_
    best_score = tune_model.best_score_
    y_pred = best_model.predict(X_test)
    
    # Printing results
    print("Best parameters:", tune_model.best_params_)
    print("Cross-validated f1 weighted score on training data: {:0.4f}".format(tune_model.best_score_))
    print()

    print(classification_report(y_test, y_pred))
    
    return best_model, best_score


# -

if __name__=="__main__":

    # Variable initialization
    raw_data_path = script_dir / 'data' / 'raw'
    clean_data_path = script_dir / 'data' / 'processed'
    predicted_data_path = script_dir / 'data' / 'predicted-data'
    model_path = script_dir / 'models' / 'hard_voting_classifier_co2_fuel.pkl'
    reports = script_dir / 'reports'/ 'figures'
    

    # Read data
    fuel_df, electric_df, hybrid_df = read_data(clean_data_path)
    non_na_rating_class, na_rating_class = remove_missing_values(fuel_df, drop_smog=False)
    
    # Set X and Y variables 
    # Response variable
    Y = non_na_rating_class[['co2_rating']]

    # Dependent variables
    X = non_na_rating_class[var_list]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
    # Set up pipeline
    # Set up parameters for the model - numerical and categorical
    numeric_features =  numeric_features
    categorical_features = categorical_features

    # Use smote to balance the data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train[numeric_features], y_train)

    # Set up preprocessor
    preprocessor = preprocessor

    # Set up model pipeline
    clf1 = KNeighborsClassifier(3,)
    clf2 = SVC(gamma=2, C=1, random_state=42, probability=True)
    clf3 = RandomForestClassifier(max_depth=100, n_estimators=10, max_features=1, random_state=42)

    classifiers = {"KNN": clf1, 
                   "SVM": clf2,
                   "RFC": clf3
                }

    eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('dt', clf3)], voting='soft')
    model = Pipeline(
            steps=[("preprocessor", preprocessor), 
                    ("hard Voting", eclf1 )] #colsample  by tree, n estimators, max depth
                                                                        )
    fig, report, score_train, score_test, balanced_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model,"Voting")
    fig.savefig(os.path.join(reports, 'hard_voting_classifier_co2_fuel.png'))
    # save report to file
    with open(script_dir / 'reports' / f"{reports}_classification_report.txt", "w") as f:
        f.write(f"Classification report for Voting Classifier (KNN, SVC, RFC):\n")
        f.write(f"Training score: {score_train}\n")
        f.write(f"Testing score: {score_test}\n")
        f.write(f"Balanced accuracy score: {balanced_accuracy}\n")
        f.write(report)
    


    params = {}
    best_dtc, dtc_score = classify_grid_search_cv_tuning(model, params, X_train, X_test, y_train, y_test, n_folds=10, scoring='f1_weighted')


    # Save model
    joblib.dump(best_dtc, model_path)
