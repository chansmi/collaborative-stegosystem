from .simple import SimpleEnvironment
from .stock_market_environment import StockMarketEnvironment

class EnvironmentFactory:
    @staticmethod
    def create_environment(environment_name, config):
        if environment_name.lower() == 'simple':
            return SimpleEnvironment(config)
        elif environment_name.lower() == 'stock_market':
            return StockMarketEnvironment(config)
        else:
            raise ValueError(f"Unknown environment: {environment_name}")