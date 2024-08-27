
import pandas as pd

def load_stock_data(config):
    df = pd.read_csv(config['data']['path'])
    return df