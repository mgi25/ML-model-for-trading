import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# MT5 + RL Libraries
try:
    import MetaTrader5 as mt5
except ImportError:
    print("Please install MetaTrader5 (pip install MetaTrader5).")
    sys.exit(1)

try:
    import gym
    from gym import spaces
except ImportError:
    print("Please install gym (pip install gym).")
    sys.exit(1)

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Please install stable-baselines3 and torch (pip install stable-baselines3 torch).")
    sys.exit(1)

# ---------------- GLOBAL CONFIG ----------------
SYMBOL = "XAUUSDm"  # Symbol to trade
INITIAL_BALANCE = 100_000.0
MAX_DAILY_DRAWDOWN_PERCENT = 4.0
MAX_OVERALL_DRAWDOWN_PERCENT = 10.0
PARTIAL_CLOSE_RATIO = 0.5
TRAIN_TIMEFRAMES_DAYS = 365   # 1 year of data
TOTAL_TRAINING_TIMESTEPS = 500_000
REWARD_MULTIPLIER = 10.0  # Scale rewards to encourage aggressive profit-taking

# ------------- Technical Indicator Functions ---------------------
def compute_rsi(series, period=14):
    """
    Compute RSI (Relative Strength Index) on a price series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD (Moving Average Convergence Divergence) for a price series.
    Returns the MACD line, signal line, and histogram.
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr(df, period=14):
    """
    Compute ATR (Average True Range) using high, low, and close prices from a dataframe.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.fillna(method='bfill')

def dynamic_lot_size(current_balance, atr_value, risk_factor=0.01, factor=100):
    """
    Determine lot size dynamically based on current balance and ATR value.
    risk_factor: fraction of the balance to risk per trade.
    factor: scaling constant to calibrate position size.
    Ensures a minimum of 1 lot and a maximum of 5 lots.
    """
    risk_amount = risk_factor * current_balance
    lots = risk_amount / (atr_value * factor)
    lots = max(1.0, lots)
    lots = min(lots, 5.0)
    return lots

# ------------- STEP 1: CONNECT TO MT5 -----------
def initialize_mt5(account_id=None, password=None, server=None):
    """
    Initialize MetaTrader5 connection.
    """
    if not mt5.initialize():
        print("MT5 Initialize() failed. Error code =", mt5.last_error())
        return False
    
    # Uncomment and fill in account details if needed:
    # if account_id and password and server:
    #     authorized = mt5.login(account_id, password=password, server=server)
    #     if not authorized:
    #         print("Failed to login to MT5 account. Error code =", mt5.last_error())
    #         return False
    return True

# ------------- STEP 2: DATA COLLECTION ----------
def download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS, num_bars=0):
    """
    Download historical data from MT5 for the given symbol/timeframe.
    """
    if num_bars <= 0:
        utc_now = datetime.now(timezone.utc)
        utc_from = utc_now - timedelta(days=start_days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
    else:
        utc_now = datetime.now(timezone.utc)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} at timeframe {timeframe}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume':'volume'}, inplace=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------- STEP 3: CUSTOM GYM ENVIRONMENT ----------
class GoldTradingEnv(gym.Env):
    """
    A refined environment for RL-based gold trading on XAUUSDM.
    
    New Features:
      - Logs each step (trade_log) for in-depth analysis.
      - Reward is based on incremental equity change scaled by a reward multiplier.
      - Additional ATR indicator included in observations.
      - Dynamic position sizing based on ATR and account balance.
    
    Observations (13 features):
      [open, high, low, close, volume, rsi, macd_hist, atr, position_size,
       current_balance, daily_drawdown%, overall_peak_balance, partial_close_flag]
    
    Actions (discrete):
      0 = hold
      1 = open long
      2 = open short
      3 = close partial (50%)
      4 = close full
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=INITIAL_BALANCE, verbose=False):
        super(GoldTradingEnv, self).__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.verbose = verbose

        # Compute technical indicators
        self.df['rsi'] = compute_rsi(self.df['close'], period=14)
        macd_line, signal_line, macd_hist = compute_macd(self.df['close'])
        self.df['macd_hist'] = macd_hist.fillna(0)
        self.df['atr'] = compute_atr(self.df, period=14)
        
        # Step pointer and account attributes
        self.current_step = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.daily_peak_balance = initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = initial_balance
        
        # For partial close tracking
        self.partial_close_triggered = False
        
        # For logging trade activity
        self.trade_log = []          # List of dicts for step-by-step logs
        self.prev_equity = self.equity  # For incremental reward calculation

        # Define observation and action space (13 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(5)
    
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            row.rsi,
            row.macd_hist,
            row.atr,
            self.position_size,
            self.current_balance,
            self.daily_drawdown_percent,
            self.overall_peak_balance,
            float(self.partial_close_triggered)
        ], dtype=np.float32)
        return obs
    
    def reset(self):
        self.current_step = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.daily_peak_balance = self.initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = self.initial_balance
        self.partial_close_triggered = False
        
        self.trade_log = []
        self.prev_equity = self.equity
        
        return self._get_observation()
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step].close
        
        # Record previous state for reward calculation
        old_equity = self.equity
        old_balance = self.current_balance
        old_position = self.position_size
        
        # Update equity based on current market price
        self._update_equity(current_price)

        # Execute the selected action
        self._execute_action(action, current_price)

        # Advance to next step
        self.current_step += 1
        done = False
        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            # Update equity after the new candle
            next_price = self.df.iloc[self.current_step].close
            self._update_equity(next_price)
        
        # Reward: incremental equity change scaled by REWARD_MULTIPLIER
        reward = ((self.equity - self.prev_equity) / self.initial_balance) * REWARD_MULTIPLIER
        self.prev_equity = self.equity

        # Apply penalties for drawdown violations
        if self._calculate_daily_drawdown() > MAX_DAILY_DRAWDOWN_PERCENT:
            reward -= 1.0  # daily drawdown penalty
            done = True
        
        if self._calculate_overall_drawdown() > MAX_OVERALL_DRAWDOWN_PERCENT:
            reward -= 2.0  # overall drawdown penalty
            done = True

        # Log step details for analysis
        step_log = {
            "step": self.current_step,
            "action": action,
            "old_position_size": old_position,
            "new_position_size": self.position_size,
            "price": current_price,
            "old_equity": old_equity,
            "new_equity": self.equity,
            "old_balance": old_balance,
            "new_balance": self.current_balance,
            "reward": reward,
            "daily_dd%": self.daily_drawdown_percent,
            "overall_dd%": self._calculate_overall_drawdown()
        }
        self.trade_log.append(step_log)

        if self.verbose:
            print(f"[Step {self.current_step}] Action={action}, Price={current_price:.2f}, "
                  f"Equity={self.equity:.2f}, Reward={reward:.4f}, PosSize={self.position_size}")

        obs = self._get_observation()
        info = {}
        
        return obs, reward, done, info

    def _execute_action(self, action, current_price):
        """
        Execute the chosen action: hold, open long/short, close partial/full.
        """
        if action == 0:
            # Hold
            pass
        elif action == 1:
            # Open long: if no position, open a new long position using dynamic lot size.
            if self.position_size == 0:
                atr_value = self.df.iloc[self.current_step].atr
                lots = dynamic_lot_size(self.current_balance, atr_value)
                self._open_position(lots=lots, direction="long", price=current_price)
        elif action == 2:
            # Open short: if no position, open a new short position using dynamic lot size.
            if self.position_size == 0:
                atr_value = self.df.iloc[self.current_step].atr
                lots = dynamic_lot_size(self.current_balance, atr_value)
                self._open_position(lots=lots, direction="short", price=current_price)
        elif action == 3:
            # Partial close (50% of position)
            if self.position_size != 0 and not self.partial_close_triggered:
                self._close_partial(current_price)
        elif action == 4:
            # Close full position
            if self.position_size != 0:
                self._close_full(current_price)
    
    def _update_equity(self, price):
        """
        Update the equity based on the current price and any open positions.
        """
        if self.position_size == 0:
            self.equity = self.current_balance
        else:
            pip_value = self.position_size * (price - self.entry_price)
            self.equity = self.current_balance + pip_value
        
        # Update daily and overall peaks
        if self.equity > self.daily_peak_balance:
            self.daily_peak_balance = self.equity
        if self.equity > self.overall_peak_balance:
            self.overall_peak_balance = self.equity
        
        # Recompute daily drawdown percentage
        self.daily_drawdown_percent = self._calculate_daily_drawdown()
    
    def _open_position(self, lots, direction, price):
        """
        Open a new position in the specified direction.
        """
        if direction == "long":
            self.position_size = lots
        else:
            self.position_size = -lots
        self.entry_price = price
    
    def _close_partial(self, current_price):
        """
        Close a partial portion (50%) of the current position.
        """
        closed_size = self.position_size * PARTIAL_CLOSE_RATIO
        remaining_size = self.position_size - closed_size
        
        pip_value = (current_price - self.entry_price)
        if self.position_size < 0:
            pip_value = (self.entry_price - current_price)
        
        realized_pnl = pip_value * closed_size
        self.current_balance += realized_pnl
        
        self.position_size = remaining_size
        if abs(self.position_size) < 1e-6:
            self.position_size = 0.0
        
        self.partial_close_triggered = True
    
    def _close_full(self, current_price):
        """
        Fully close the current open position.
        """
        if self.position_size == 0:
            return
        pip_value = (current_price - self.entry_price)
        if self.position_size < 0:
            pip_value = (self.entry_price - current_price)

        realized_pnl = pip_value * abs(self.position_size)
        self.current_balance += realized_pnl
        
        self.position_size = 0.0
        self.entry_price = 0.0
        self.partial_close_triggered = False

    def _calculate_daily_drawdown(self):
        if self.daily_peak_balance == 0:
            return 0.0
        dd = (self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0
        return max(dd, 0.0)
    
    def _calculate_overall_drawdown(self):
        if self.overall_peak_balance == 0:
            return 0.0
        dd = (self.overall_peak_balance - self.equity) / self.overall_peak_balance * 100.0
        return max(dd, 0.0)
    
    def render(self, mode='human'):
        pass  # Optional: Implement visualization if needed

# -------------------- MAIN SCRIPT ------------------
def main():
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training using device: {device}")

    # 1) Initialize MT5 connection
    if not initialize_mt5():
        print("Failed to initialize MT5.")
        return
    
    # 2) Download historical data
    df = download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS)
    if df.empty:
        print("No data retrieved, exiting.")
        return
    
    # Split data into training and testing datasets
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    
    # 3) Create training environment
    train_env = GoldTradingEnv(df_train, initial_balance=INITIAL_BALANCE, verbose=False)
    vec_train_env = DummyVecEnv([lambda: train_env])
    
    # 4) Train the RL model using PPO with enhanced settings and GPU support
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = PPO(
        "MlpPolicy",
        vec_train_env,
        verbose=1,
        tensorboard_log="./ppo_xauusdm_tensorboard/",
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device=device  # Use GPU if available
    )
    print(f"Starting PPO training for {TOTAL_TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS)
    print("Training finished!")
    
    # 5) Test the trained model (set verbose=True for detailed logs)
    test_env = GoldTradingEnv(df_test, initial_balance=INITIAL_BALANCE, verbose=True)
    obs = test_env.reset()
    
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
    
    # Print final test results
    print(f"Test completed. Total reward: {total_reward:.2f}")
    print(f"Final balance: {test_env.current_balance:.2f}")
    print(f"Max daily drawdown in test: {test_env.daily_drawdown_percent:.2f}%")
    
    # Save trade log to CSV for further analysis
    trade_log_df = pd.DataFrame(test_env.trade_log)
    trade_log_df.to_csv("test_trade_log.csv", index=False)
    print("Saved test trade log to test_trade_log.csv")
    
    # 6) Save the trained model
    model.save("ppo_gold_trader_xauusdm")
    
    # 7) Shut down the MT5 connection
    mt5.shutdown()

if __name__ == "__main__":
    main()
