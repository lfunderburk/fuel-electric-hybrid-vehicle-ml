# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

import pandas as pd
import sys, os
import joblib
import utils
from sklearn.impute import KNNImputer
from dotenv import load_dotenv
from pathlib import Path
# +
def predict_co2_rating(df, list_of_vars, model):
    # Preprocess the original data (fuel_df)
    X_fuel = df[list_of_vars].copy()
    target = df['original_co2r'].copy()

    # Get remaining columns from df
    remaining_cols = [col for col in df.columns if col not in list_of_vars and col != 'original_co2r']
    remaining_df = df[remaining_cols].copy()

    # Predict missing "co2_rating" values
    X_fuel['predicted_co2_rating'] = model.predict(X_fuel)

    # Merge the predicted values with the original data
    fuel_df_pred = pd.concat([X_fuel, target, remaining_df], axis=1)

    return fuel_df_pred

# +
def impute_data(numeric_features, df, target):

    df_c = df.copy()
    # Independent variables
    X = df_c[numeric_features + [target]]

    missing_values_index = df_c[df_c.isnull().any(axis=1)].index

    imputer = KNNImputer(n_neighbors=1)
    X_imputed = imputer.fit_transform(X)

    X_imputed[missing_values_index, -1]

    df_c.loc[missing_values_index, target] =  X_imputed[missing_values_index, -1]

    return df_c


# -

if __name__=="__main__":

    load_dotenv()  # load environment variables from .env file
    PROJECT_DIR = os.getenv('PROJECT_DIR')


    # Variable initialization
    raw_data_path = os.path.join(PROJECT_DIR, 'data', 'raw')
    clean_data_path = os.path.join(PROJECT_DIR, 'data', 'processed')
    predicted_data_path = os.path.join(PROJECT_DIR, 'data', 'predicted-data')
    model_path = os.path.join(PROJECT_DIR, 'models', 'hard_voting_classifier_co2_fuel.pkl')

    # Load data
    fuel_df, electric_df, hybrid_df = utils.read_data(clean_data_path)
    
    non_na_rating_class, na_rating_class = utils.remove_missing_values(fuel_df, drop_smog=False)
    non_na_rating_class.rename(columns={'co2_rating': 'original_co2r'}, inplace=True)
    na_rating_class.rename(columns={'co2_rating': 'original_co2r'}, inplace=True)

    # Load model
    best_dtc = joblib.load(model_path)
    
    # Use model to make predictions
    non_na_pred = predict_co2_rating(non_na_rating_class, utils.var_list, best_dtc)
    na_pred = predict_co2_rating(na_rating_class, utils.var_list, best_dtc)

    # Merge the predicted values with the original data
    fuel_df_pred = pd.concat([non_na_pred, na_pred], axis=0)

    # Impute smog_rating 
    fuel_df_pred = impute_data(utils.numeric_features, fuel_df_pred, 'smog_rating')
    
    # Save the data
    fuel_df_pred.to_csv(Path(predicted_data_path,'predicted_co2_rating.csv'), index=False)

    # Predict missing "co2_rating" values in hybrid_df
    hybrid_df_pred = impute_data(utils.numeric_features, hybrid_df, 'co2_rating')

    # Predict missing 'smog_rating' values in hybrid_df
    hybrid_df_pred = impute_data(utils.numeric_features, hybrid_df_pred, 'smog_rating')

    # Save the data
    hybrid_df_pred.to_csv(Path(predicted_data_path,'predicted_co2_rating_hybrid.csv'), index=False)

    num_e = ['model_year','consumption_city(kwh/100km)',
            'fuelconsumption_hwy(kwh/100km)', 'fuelconsumption_comb(kwh/100km)',
            'fuelconsumption_city(le/100km)', 'fuelconsumption_hwy(le/100km)',
            'fuelconsumption_comb(le/100km)','recharge_time(h)',
            'co2emissions_(g/km)']

    # Predict missing "co2_rating" values in electric_df
    electric_df_pred = impute_data(num_e, electric_df, 'co2_rating')

    # Predict missing 'smog_rating' values in electric_df
    electric_df_pred = impute_data(num_e, electric_df_pred, 'smog_rating')

    # Save the data
    electric_df_pred.to_csv(Path(predicted_data_path, 'predicted_co2_rating_electric.csv'), index=False)
