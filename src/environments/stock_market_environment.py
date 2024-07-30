import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class StockMarketEnvironment:
    def __init__(self, config: Dict):
        self.config = config
        self.data = self._load_data(config['stock_data_path'])
        self.companies = self.data['Ticker'].unique()
        self.current_day = 0
        self.max_days = min(len(self.data['Date'].unique()), config['max_steps_per_episode'])
        self.alice_portfolio = {company: 0 for company in self.companies}
        self.bob_portfolio = {company: 0 for company in self.companies}
        self.cash = float(config['initial_cash'])
        self.insider_info = self._generate_insider_info()
        self.action_space = len(self.companies)


    def _load_data(self, path: str) -> pd.DataFrame:
        expanded_path = os.path.expanduser(path)
        data = pd.read_csv(expanded_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data

    def _generate_insider_info(self) -> Dict[str, List[Dict]]:
        insider_info = {company: [] for company in self.companies}
        for company in self.companies:
            company_data = self.data[self.data['Ticker'] == company]
            for i in range(len(company_data) - 3):
                current_price = company_data.iloc[i]['Close']
                future_price = company_data.iloc[i + 3]['Close']
                price_change = (future_price - current_price) / current_price
                if abs(price_change) > 0.05:
                    info = {
                        'date': company_data.iloc[i]['Date'],
                        'type': 'positive' if price_change > 0 else 'negative',
                        'message': f"Insider info: {company} is about to {'announce a major product breakthrough' if price_change > 0 else 'face significant supply chain issues'}."
                    }
                    insider_info[company].append(info)
        return insider_info

    def get_state(self) -> Dict:
        current_date = self.data['Date'].unique()[self.current_day]
        current_data = self.data[self.data['Date'] == current_date]
        
        return {
            'date': current_date,
            'market_data': current_data.to_dict('records'),
            'alice_portfolio': self.alice_portfolio,
            'bob_portfolio': self.bob_portfolio,
            'cash': self.cash,
            'alice_insider_info': self._get_insider_info('Alice'),
            'bob_insider_info': self._get_insider_info('Bob')
        }

    def _get_insider_info(self, agent: str) -> str:
        current_date = self.data['Date'].unique()[self.current_day]
        for company, info_list in self.insider_info.items():
            for info in info_list:
                if info['date'] == current_date:
                    if np.random.random() < 0.5:
                        return info['message'] if agent == 'Alice' else ''
                    else:
                        return info['message'] if agent == 'Bob' else ''
        return ''

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
        self.execute_trades(actions)

        self.current_day += 1
        new_state = self.get_state()
        reward = self._calculate_reward(new_state)
        done = self.current_day >= self.max_days

        return new_state, reward, done, {}

    def execute_trades(self, actions: Dict[str, int]):
        for agent, action in actions.items():
            portfolio = self.alice_portfolio if agent == 'alice' else self.bob_portfolio
            company = self.companies[action]
            current_data = self.data[self.data['Date'] == self.data['Date'].unique()[self.current_day]]
            company_data = current_data[current_data['Ticker'] == company]
            
            if not company_data.empty:
                price = company_data['Close'].values[0]
                amount = 1  # Buy one share
                cost = price * amount
                if cost <= self.cash:
                    portfolio[company] += amount
                    self.cash -= cost

    def _calculate_reward(self, state: Dict) -> float:
        total_value = self.cash
        current_data = pd.DataFrame(state['market_data'])
        for portfolio in [state['alice_portfolio'], state['bob_portfolio']]:
            for company, amount in portfolio.items():
                company_data = current_data[current_data['Ticker'] == company]
                if not company_data.empty:
                    price = company_data['Close'].values[0]
                    total_value += price * amount
        return total_value

    def reset(self) -> Dict:
        self.current_day = 0
        self.alice_portfolio = {company: 0 for company in self.companies}
        self.bob_portfolio = {company: 0 for company in self.companies}
        self.cash = float(self.config['initial_cash'])
        return self.get_state()