import pandas as pd
from src.data.data_extraction import fuel_consumption_metadata_extraction

def test_fuel_consumption_metadata_extraction():
    result = fuel_consumption_metadata_extraction()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'name' in result.columns
    assert 'url' in result.columns
    assert result['name'].notnull().all()
    assert result['url'].notnull().all()
