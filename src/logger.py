# src/logger.py

import os
import json
from datetime import datetime

class Logger:
    def __init__(self, config):
        self.config = config
        self.results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.results_dir, exist_ok=True)
        self.conversation_log = []
        self.trade_log = []
        self.portfolio_log = []

    def log_conversation(self, day, conversation, alice_inner_dialogue, bob_inner_dialogue):
        self.conversation_log.append({
            'day': day,
            'conversation': conversation,
            'alice_inner_dialogue': alice_inner_dialogue,
            'bob_inner_dialogue': bob_inner_dialogue
        })

    def log_trade(self, day, trader, stock, amount, price):
        self.trade_log.append({
            'day': day,
            'trader': trader,
            'stock': stock,
            'amount': amount,
            'price': price
        })

    def log_portfolio(self, day, alice_portfolio, bob_portfolio, alice_budget, bob_budget):
        self.portfolio_log.append({
            'day': day,
            'alice_portfolio': alice_portfolio,
            'bob_portfolio': bob_portfolio,
            'alice_budget': alice_budget,
            'bob_budget': bob_budget
        })

    def save_logs(self):
        with open(os.path.join(self.results_dir, 'conversations.json'), 'w') as f:
            json.dump(self.conversation_log, f, indent=2)

        with open(os.path.join(self.results_dir, 'trades.json'), 'w') as f:
            json.dump(self.trade_log, f, indent=2)

        with open(os.path.join(self.results_dir, 'portfolios.json'), 'w') as f:
            json.dump(self.portfolio_log, f, indent=2)