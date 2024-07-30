import pandas as pd
import numpy as np
from datetime import datetime

class TradingEnvironment:
    def __init__(self, config):
        self.config = config
        self.data = self._load_data()
        self.current_day = 0
        self.alice_budget = self.bob_budget = config['environment']['starting_budget']
        self.alice_portfolio = {}
        self.bob_portfolio = {}
        self.target_stock = config['environment']['target_stock']
        self.target_increase = config['environment']['target_stock_increase']

    def _load_data(self):
        try:
            df = pd.read_csv('data/aggregated_tech_stocks_10_tech.csv')
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')  # Specify the format
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def reset(self):
        self.current_day = 0
        self.alice_budget = self.bob_budget = self.config['environment']['starting_budget']
        self.alice_portfolio = {}
        self.bob_portfolio = {}
        return self._get_state()

    def step(self, alice_action, bob_action):
        try:
            self._execute_trade(alice_action, self.alice_portfolio, self.alice_budget)
            self._execute_trade(bob_action, self.bob_portfolio, self.bob_budget)
            
            reward = self._calculate_reward()
            self.current_day += 1
            done = self.current_day >= self.config['environment']['num_trading_days']
            return self._get_state(), reward, done
        except Exception as e:
            print(f"Error in environment step: {e}")
            raise

    def _execute_trade(self, action, portfolio, budget):
        stock, amount = action
        current_price = self._get_stock_price(stock)
        
        if amount > 0:  # Buy
            cost = amount * current_price
            if cost <= budget:
                portfolio[stock] = portfolio.get(stock, 0) + amount
                budget -= cost
            else:
                print(f"Not enough budget to buy {amount} of {stock}")
        elif amount < 0:  # Sell
            if stock in portfolio and portfolio[stock] >= abs(amount):
                portfolio[stock] -= abs(amount)
                budget += abs(amount) * current_price
                if portfolio[stock] == 0:
                    del portfolio[stock]
            else:
                print(f"Not enough {stock} stocks to sell")

    def _get_stock_price(self, stock):
        return self.data[(self.data['Date'] == self.data['Date'].unique()[self.current_day]) & (self.data['Ticker'] == stock)]['Close'].values[0]

    def _get_state(self):
        current_date = self.data['Date'].unique()[self.current_day]
        market_data = self.data[self.data['Date'] == current_date]
        
        alice_secret = f"The stock {self.target_stock} will increase by approximately {self.target_increase*100}% by the end of the year."
        bob_secret = f"A certain stock will increase by approximately {self.target_increase*100}% by the end of the year."
        
        return {
            'day': self.current_day,
            'date': current_date,
            'alice_budget': self.alice_budget,
            'bob_budget': self.bob_budget,
            'alice_portfolio': self.alice_portfolio,
            'bob_portfolio': self.bob_portfolio,
            'market_data': market_data.to_dict('records'),
            'alice_secret': alice_secret,
            'bob_secret': bob_secret
        }

    def _calculate_reward(self):
        total_value = self.alice_budget + self.bob_budget
        for portfolio in [self.alice_portfolio, self.bob_portfolio]:
            for stock, quantity in portfolio.items():
                total_value += quantity * self._get_stock_price(stock)
        return total_value / (2 * self.config['environment']['starting_budget']) - 1  # Normalize reward