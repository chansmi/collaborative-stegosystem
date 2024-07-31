# environment.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingEnvironment:
    def __init__(self, config):
        self.config = config
        self.data = self._load_data()
        self.reset()

    def _load_data(self):
        df = pd.read_csv('data/aggregated_tech_stocks_10_tech.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        return df

    def reset(self):
        self.current_day = 0
        self.alice_budget = self.bob_budget = self.config['environment']['starting_budget']
        self.alice_portfolio = {ticker: 0 for ticker in self.data['Ticker'].unique()}
        self.bob_portfolio = {ticker: 0 for ticker in self.data['Ticker'].unique()}
        self.target_stock = self.config['environment']['target_stock']
        self.target_increase = self.config['environment']['target_stock_increase']
        return self._get_state()

    def step(self, alice_actions, bob_actions):
        for action in alice_actions:
            self._execute_trade(action, self.alice_portfolio, 'alice_budget')
        for action in bob_actions:
            self._execute_trade(action, self.bob_portfolio, 'bob_budget')
        
        reward = self._calculate_reward()
        self.current_day += 1
        done = self.current_day >= self.config['environment']['num_trading_days']
        return self._get_state(), reward, done

    def _execute_trade(self, action, portfolio, budget_attr):
        stock, amount = action
        if stock is None or amount == 0:
            return

        current_price = self._get_stock_price(stock)
        
        if amount > 0:  # Buy
            cost = amount * current_price
            if cost <= getattr(self, budget_attr):
                portfolio[stock] += amount
                setattr(self, budget_attr, getattr(self, budget_attr) - cost)
        elif amount < 0:  # Sell
            if portfolio[stock] >= abs(amount):
                portfolio[stock] -= abs(amount)
                setattr(self, budget_attr, getattr(self, budget_attr) + abs(amount) * current_price)

    def _get_stock_price(self, stock):
        current_date = self.data['Date'].unique()[self.current_day]
        return self.data[(self.data['Date'] == current_date) & (self.data['Ticker'] == stock)]['Close'].values[0]

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