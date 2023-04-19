import requests
import os
import tempfile
import pandas as pd
from src.data.data_extraction import extract_raw_data, save_raw_data, read_and_clean_csv_file, \
    fuel_consumption_metadata_extraction

def test_fuel_consumption_metadata_extraction():
    result = fuel_consumption_metadata_extraction()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'name' in result.columns
    assert 'url' in result.columns
    assert result['name'].notnull().all()
    assert result['url'].notnull().all()

def test_extract_raw_data():
    test_url = "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY2022%20Fuel%20Consumption%20Ratings.csv"
    result = extract_raw_data(test_url)

    assert isinstance(result, requests.models.Response)
    assert result.status_code == 200
    assert len(result.text) > 0


def test_extract_raw_data_invalid_url():
    test_url = "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY2022%20Fuel%20Consumption%20Ratings.c"
    result = extract_raw_data(test_url)

    assert isinstance(result, requests.models.Response)
    assert result.status_code == 404

def test_save_raw_data():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder_path = tmpdir
        test_url_content = requests.models.Response()
        test_url_content._content = b'Model,Make,Model,Vehicle Class,Engine Size,Cylinders,Transmission,Fuel,Fuel Consumption,,,,CO2 Emissions,CO2,Smog,,,,\r\n'

        test_file_name = "output.csv"
        save_raw_data(test_folder_path, test_url_content, test_file_name)

        # Check if the file is saved in the temporary directory
        file_path = os.path.join(test_folder_path, test_file_name)
        assert os.path.isfile(file_path)

        # Check if the content of the saved file is correct
        with open(file_path, "rb") as f:
            content = f.read()
            assert content == test_url_content.content

def test_read_and_clean_csv_file():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a sample CSV file
        test_csv_content = (
            'Model,Make,Model,Vehicle Class,Engine Size,Cylinders,Transmission,Fuel,Fuel Consumption,,,,CO2 Emissions,CO2,Smog,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\r\n'
            'Year,,,,(L),,,Type,City (L/100 km),Hwy (L/100 km),Comb (L/100 km),Comb (mpg),(g/km),Rating,Rating,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\r\n'
            '2023,Acura,Integra,Full-size,1.5,4,AV7,Z,7.9,6.3,7.2,39,167,6,7,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\r\n'
            '2023,Acura,Integra A-SPEC,Full-size,1.5,4,AV7,Z,8.1,6.5,7.4,38,172,6,7,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
        )
        test_csv_path = os.path.join(tmpdir, 'test_data.csv')
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)

        # Test read_and_clean_csv_file function
        result = read_and_clean_csv_file(tmpdir, 'test_data.csv')

        # Check if the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check the shape of the DataFrame
        assert result.shape == (2, 17)

        # Check the column names of the DataFrame
        expected_columns = [
            'model_year', 'make_', 'model.1_', 'vehicleclass_', 'enginesize_(l)', 'cylinders_', 'transmission_', 'fuel_type',
            'fuelconsumption_city(l/100km)', 'fuelconsumption_hwy(l/100km)', 'fuelconsumption_comb(l/100km)', 'fuelconsumption_comb(mpg)',
            'co2emissions_(g/km)', 'co2_rating', 'smog_rating', 'transmission_type', 'number_of_gears'
        ]
        assert list(result.columns) == expected_columns