import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Var list
var_list = ['vehicleclass_','make_',
                    'model.1_','model_year',
                    'cylinders_','fuelconsumption_city(l/100km)',
                    'fuelconsumption_hwy(l/100km)',
                    'fuelconsumption_comb(l/100km)',
                    #'fuelconsumption_comb(mpg)',
                    'co2emissions_(g/km)',
                    'number_of_gears']

# Set up parameters for the model - numerical and categorical
numeric_features =  ['model_year','cylinders_',
                    'fuelconsumption_city(l/100km)',
                    'fuelconsumption_hwy(l/100km)',
                    'fuelconsumption_comb(l/100km)',
                    #'fuelconsumption_comb(mpg)',
                    'co2emissions_(g/km)','number_of_gears']
categorical_features = ['vehicleclass_']

# Set up numerical and categorical transformers
numeric_transformer = Pipeline(
                            steps=[("scaler", StandardScaler())]
                        )

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Set up preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        #("cat", categorical_transformer, categorical_features),
    ]
)

def remove_missing_values(fuel_df, drop_smog=True, rating_column='co2_rating', drop_column='smog_rating'):

    fuel_df_copy = fuel_df.copy()

    # Drop smog_rating from non_na_rating
    if drop_smog:
        fuel_df_copy.drop(columns=[drop_column], inplace=True)
    else:
        pass
    fuel_df_copy['number_of_gears'].fillna(0, inplace=True)

    # Set up data pipeline - goal is to predict co2_rating 
    na_rating = fuel_df_copy[fuel_df_copy[rating_column].isna()]
    non_na_rating = fuel_df_copy[~fuel_df_copy[rating_column].isna()]

    non_na_rating_class = non_na_rating.copy()
    na_rating_class = na_rating.copy()

    non_na_rating_class[rating_column] = non_na_rating_class[rating_column].astype(int)

    return non_na_rating_class, na_rating_class

def read_data(path):
    """
    This function reads data from csv files

    Parameters:
    ----------
        path: str
            path to data files

    Returns:
    -------
        fuel_df: pandas.DataFrame
            dataframe containing fuel cars data
        electric_df: pandas.DataFrame
            dataframe containing electric cars data
        hybrid_df: pandas.DataFrame
            dataframe containing hybrid cars data

    """
    
    # Fuel based cars
    file_name_2022_1995 = "1995_today_vehicle_fuel_consumption.csv"
    
    # Electric cars
    pure_electric = "Batteryelectric_vehicles__.csv"
    hybric_vehicle = "Plugin_hybrid_electric_vehicles__.csv"

    # Read data files
    fuel_df = pd.read_csv(Path(path ,f'{file_name_2022_1995}'))
    electric_df = pd.read_csv(Path(path ,f'{pure_electric}'))
    hybrid_df = pd.read_csv(Path(path ,f'{hybric_vehicle}'))

    return fuel_df, electric_df, hybrid_df