import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_stock_data():
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(5)]
    
    data = []
    
    for date in dates:
        for stock in stocks:
            if stock == 'AAPL':
                # Make AAPL more volatile and generally increasing
                open_price = np.random.uniform(150, 160)
                close_price = open_price * np.random.uniform(0.98, 1.05)
            else:
                # Other stocks have less movement
                open_price = np.random.uniform(100, 200)
                close_price = open_price * np.random.uniform(0.99, 1.01)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': stock,
                'Open': round(open_price, 2),
                'Close': round(close_price, 2),
                'Volume': int(np.random.uniform(1000000, 5000000))
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/stock_data.csv', index=False)
    print("Stock data generated and saved to data/stock_data.csv")

if __name__ == "__main__":
    generate_stock_data()