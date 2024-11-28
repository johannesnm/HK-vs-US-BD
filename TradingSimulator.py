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
        self.avg_buy_price = 0  # To keep track of the average price of positions

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0
        self.trade_log = []
        self.avg_buy_price = 0

    def buy(self, price, quantity):
        total_cost = price * quantity * (1 + self.transaction_fee)
        if self.cash >= total_cost:
            # Update average buy price if already holding positions
            if self.position > 0:
                self.avg_buy_price = ((self.avg_buy_price * self.position) + (price * quantity)) / (self.position + quantity)
            else:
                self.avg_buy_price = price

            self.cash -= total_cost
            self.position += quantity
            self.trade_log.append(('BUY', price, quantity, 0, self.get_portfolio_value(price)))
        else:
            print(f"BUY FAILED: Not enough cash (Available: {self.cash:.2f}, Needed: {total_cost:.2f})")

    def sell(self, price, quantity):
        if self.position >= quantity:
            total_revenue = price * quantity * (1 - self.transaction_fee)
            # Calculate profit or loss based on avg buy price
            profit_loss = (price - self.avg_buy_price) * quantity - (price * quantity * self.transaction_fee)

            self.cash += total_revenue
            self.position -= quantity
            self.trade_log.append(('SELL', price, quantity, profit_loss, self.get_portfolio_value(price)))

            # Reset average price if all positions are sold
            if self.position == 0:
                self.avg_buy_price = 0
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
