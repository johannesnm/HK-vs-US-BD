import matplotlib.pyplot as plt
import pandas as pd

class TradingSimulator:
    """A Trading simulator to evaluate the arbitrage trading performance between HK and US markets."""

    def __init__(self, initial_cash=100000, transaction_fee=0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.transaction_fee = transaction_fee
        self.trade_log = []
        self.avg_buy_price = 0  # To keep track of the average price of positions

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0
        self.trade_log = []
        self.avg_buy_price = 0

    def buy(self, hk_close_price):
        quantity = 100  # Fixed number of shares to buy
        total_cost = hk_close_price * quantity * (1 + self.transaction_fee)

        if self.cash >= total_cost:
            # Update the average buy price if already holding positions
            if self.position > 0:
                self.avg_buy_price = ((self.avg_buy_price * self.position) + (hk_close_price * quantity)) / (self.position + quantity)
            else:
                self.avg_buy_price = hk_close_price

            self.cash -= total_cost
            self.position += quantity
            self.trade_log.append(('BUY', hk_close_price, quantity, 0, self.get_portfolio_value(hk_close_price)))
        else:
            print(f"BUY FAILED: Not enough cash (Available: {self.cash:.2f}, Needed: {total_cost:.2f})")

    def sell(self, us_open_price):
        quantity = 100  # Fixed number of shares to sell

        if self.position >= quantity:
            total_revenue = us_open_price * quantity * (1 - self.transaction_fee)
            # Calculate profit or loss based on average buy price
            profit_loss = (us_open_price - self.avg_buy_price) * quantity - (us_open_price * quantity * self.transaction_fee)

            self.cash += total_revenue
            self.position -= quantity
            self.trade_log.append(('SELL', us_open_price, quantity, profit_loss, self.get_portfolio_value(us_open_price)))

            # Reset average price if all positions are sold
            if self.position == 0:
                self.avg_buy_price = 0
        else:
            print(f"SELL FAILED: Not enough position to sell (Current Position: {self.position})")

    def run(self, df, hk_close_column='HK_Close', us_open_column='US_Open', signal_column='signal'):
        """
        Run the simulator on the given DataFrame using the provided signals.
        
        The buy action will occur on the HK close price, and the corresponding sell action will occur
        at the US open price.
        """
        for index, row in df.iterrows():
            signal = row[signal_column]
            hk_close_price = row[hk_close_column]
            us_open_price = row[us_open_column]

            if signal == 1:  # Buy signal
                self.buy(hk_close_price)  # Buy at HK close price
            elif signal == -1 and self.position > 0:  # Sell signal and we hold position
                self.sell(us_open_price)  # Sell at US open price

    def get_portfolio_value(self, current_price):
        return self.cash + (self.position * current_price)

    def get_trade_log(self):
        """Returns the trade logs in a Pandas DataFrame."""
        return pd.DataFrame(self.trade_log, columns=['Action', 'Price', 'Quantity', 'Profit/Loss', 'Portfolio Value'])

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

    def plot_trade_outcomes(self):
        """Plots the distribution of profitable and losing trades."""
        trade_log_df = self.get_trade_log()
        if trade_log_df.empty:
            print("No trades executed. Cannot plot trade outcomes.")
            return

        # Separate profitable and losing trades
        profitable_trades = trade_log_df[trade_log_df['Profit/Loss'] > 0]
        losing_trades = trade_log_df[trade_log_df['Profit/Loss'] < 0]

        # Plot histograms
        plt.figure(figsize=(12, 6))
        if not profitable_trades.empty:
            plt.hist(profitable_trades['Profit/Loss'], bins=20, alpha=0.7, label='Profitable Trades', color='green')
        if not losing_trades.empty:
            plt.hist(losing_trades['Profit/Loss'], bins=20, alpha=0.7, label='Losing Trades', color='red')
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.title("Distribution of Trade Outcomes")
        plt.xlabel("Profit/Loss")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

    def calculate_metrics(self):
        """Calculates various metrics to evaluate the model performance."""
        trade_log_df = self.get_trade_log()
        if trade_log_df.empty:
            print("No trades executed. No metrics available.")
            return

        # Filter only SELL actions to evaluate win and loss trades
        sell_trades = trade_log_df[trade_log_df['Action'] == 'SELL']

        total_profit = sell_trades['Profit/Loss'].sum()
        total_trades = len(sell_trades)
        win_trades = len(sell_trades[sell_trades['Profit/Loss'] > 0])
        loss_trades = len(sell_trades[sell_trades['Profit/Loss'] < 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        print("\n=== Performance Metrics ===")
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print("----------------------------\n")

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

    
