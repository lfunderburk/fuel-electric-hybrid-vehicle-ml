import requests
from src.data.data_extraction import extract_raw_data
import os
import tempfile
from src.data.data_extraction import save_raw_data

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


