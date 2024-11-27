import matplotlib.pyplot as plt
import pandas as pd

class TradingSimulator:
    """A Trading simulator to evaluate the trading performance of models."""

    def __init__(self, initial_cash=100000, transaction_fee=0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.transaction_fee = transaction_fee
        self.trade_log = []

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0
        self.trade_log = []

    def buy(self, price, quantity):
        total_cost = price * quantity * (1 + self.transaction_fee)
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.position += quantity
            self.trade_log.append(('BUY', price, quantity, total_cost, self.get_portfolio_value(price)))
        else:
            print(f"BUY FAILED: Not enough cash (Available: {self.cash:.2f}, Needed: {total_cost:.2f})")

    def sell(self, price, quantity):
        if self.position >= quantity:
            total_revenue = price * quantity * (1 - self.transaction_fee)
            self.cash += total_revenue
            self.position -= quantity
            self.trade_log.append(('SELL', price, quantity, total_revenue, self.get_portfolio_value(price)))
        else:
            print(f"SELL FAILED: No position to sell (Current Position: {self.position})")

    def run(self, df, signal_column='signal', price_column='US_Close'):
        skipped_sell_count = 0
        for index, row in df.iterrows():
            signal = row[signal_column]
            price = row[price_column]

            if signal == 1:  # Buy signal
                self.buy(price, 10)  # Buy 10 units arbitrarily
            elif signal == -1:  # Sell signal
                if self.position > 0:
                    self.sell(price, 10)
                else:
                    skipped_sell_count += 1

        if skipped_sell_count > 0:
            print(f"Skipped {skipped_sell_count} sell signals due to insufficient positions.")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.position * current_price)

    def get_trade_log(self):
        """Returns the trade logs in a Pandas DataFrame."""
        return pd.DataFrame(self.trade_log, columns=['Action', 'Price', 'Quantity', 'Total Cost/Revenue', 'Portfolio Value'])

    def plot_portfolio_growth(self):
        """Plots the portfolio value over time."""
        trade_log_df = self.get_trade_log()
        if trade_log_df.empty:
            print("No trades executed. Cannot plot portfolio growth.")
            return
        
        trade_log_df['Portfolio Value'].plot(figsize=(12, 6), title="Portfolio Value Over Time", marker='o')
        plt.xlabel("Trades")
        plt.ylabel("Portfolio Value")
        plt.grid()
        plt.show()

    def summary(self, final_price):
        """Prints a summary of the trading performance."""
        trade_log_df = self.get_trade_log()
        if trade_log_df.empty:
            print("No trades executed. No performance summary available.")
            return

        final_portfolio_value = self.get_portfolio_value(final_price)
        portfolio_growth = ((final_portfolio_value - self.initial_cash) / self.initial_cash) * 100
        total_buys = len(trade_log_df[trade_log_df['Action'] == 'BUY'])
        total_sells = len(trade_log_df[trade_log_df['Action'] == 'SELL'])

        print("\n=== Trading Performance Summary ===")
        print(f"Initial Cash: {self.initial_cash:.2f}")
        print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"Portfolio Growth: {portfolio_growth:.2f}%")
        print(f"Total Buys: {total_buys}, Total Sells: {total_sells}")
        print("-----------------------------------\n")




