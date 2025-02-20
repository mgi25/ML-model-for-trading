import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ------------- MT5 and RL Libraries ---------------
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

# ------------- GLOBAL CONFIG -----------------------
SYMBOL = "XAUUSDm"
INITIAL_BALANCE = 100_000.0
MAX_DAILY_DRAWDOWN_PERCENT = 4.0
MAX_OVERALL_DRAWDOWN_PERCENT = 10.0
RISK_REDUCTION_AFTER_DD = 0.5  # e.g., once daily DD is hit, risk is halved
PARTIAL_CLOSE_RATIO = 0.5      # close 50% position at 1:1 R:R

# ------------- STEP 1: CONNECT TO MT5 -------------
def initialize_mt5(account_id=None, password=None, server=None):
    """
    Initialize MetaTrader5 connection. Update with your actual account details if needed.
    """
    if not mt5.initialize():
        print("MT5 Initialize() failed. Error code =", mt5.last_error())
        return False
    
    # If you need to log in to a specific account
    if account_id and password and server:
        authorized = mt5.login(account_id, password=password, server=server)
        if not authorized:
            print("Failed to login to MT5 account. Error code =", mt5.last_error())
            return False
    return True

# ------------- STEP 2: DATA COLLECTION -------------
def download_data(
    symbol=SYMBOL, 
    timeframe=mt5.TIMEFRAME_M5, 
    start_days=365,
    num_bars=0
):
    """
    Download historical data from MT5 for the given symbol and timeframe.
    If num_bars > 0, we fetch that many bars from 'now' backward.
    Otherwise, we fetch from 'now - start_days' until 'now'.
    """
    if num_bars <= 0:
        utc_from = datetime.utcnow() - timedelta(days=start_days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, datetime.utcnow())
    else:
        # from 'now' backward
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} at timeframe {timeframe}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume':'volume'}, inplace=True)
    df = df[['time','open','high','low','close','volume']]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------- STEP 3: GYM ENVIRONMENT -------------
class GoldTradingEnv(gym.Env):
    """
    A simplified environment for RL-based gold trading.
    Observations:
      - current candle (OHLC)
      - maybe some indicators (none in this simple example)
      - current balance, equity, floating P/L
      - how many lots currently open (and direction)
      - day’s realized drawdown
    
    Actions (Discrete for example):
      0 = hold
      1 = open long
      2 = open short
      3 = close partial (50%)
      4 = close full
    This can be extended to continuous (for position size), but we'll keep it basic here.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=INITIAL_BALANCE):
        super(GoldTradingEnv, self).__init__()
        
        # Store data, account variables, etc.
        self.df = df
        self.initial_balance = initial_balance
        
        # Current step in the data
        self.current_step = 0
        
        # Positions
        self.position_size = 0.0  # +ve = long, -ve = short
        self.entry_price = 0.0
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.daily_peak_balance = initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = initial_balance
        
        # For partial close logic
        self.partial_close_triggered = False
        
        # Define observation space (for simplicity, just 10 floats)
        # Example: [open, high, low, close, volume, position_size, current_balance, daily_drawdown_percent, ...]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Define action space: discrete 5 possible actions
        self.action_space = spaces.Discrete(5)
        
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            self.position_size,
            self.current_balance,
            self.daily_drawdown_percent,
            # Additional features:
            self.overall_peak_balance,
            float(self.partial_close_triggered)
        ], dtype=np.float32)
        return obs
    
    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.daily_peak_balance = self.initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = self.initial_balance
        self.partial_close_triggered = False
        
        return self._get_observation()
    
    def step(self, action):
        """
        Step through the environment:
         1. interpret action (open/close partial/close full/etc.)
         2. move to the next candle
         3. update balance/equity
         4. compute reward
         5. check if done
        """
        # If there's an open position, update floating P/L
        current_price = self.df.iloc[self.current_step].close
        self._update_equity(current_price)
        
        reward = 0.0
        done = False
        
        # ---- Execute the chosen action ----
        if action == 0:
            # hold
            pass
        elif action == 1:
            # open long
            if self.position_size == 0:
                self._open_position(lots=1, direction="long", price=current_price)
        elif action == 2:
            # open short
            if self.position_size == 0:
                self._open_position(lots=1, direction="short", price=current_price)
        elif action == 3:
            # partial close
            if self.position_size != 0 and not self.partial_close_triggered:
                self._close_partial(current_price)
        elif action == 4:
            # close full
            if self.position_size != 0:
                self._close_full(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # If we haven't reached the end of the data
        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            # Update equity with new candle's close
            next_price = self.df.iloc[self.current_step].close
            self._update_equity(next_price)
        
        # ---- Compute reward (simple: change in equity) ----
        # This is a simplistic approach: reward = (equity - initial_balance)/initial_balance per step
        reward = (self.equity - self.initial_balance) / self.initial_balance
        
        # ---- Check Drawdown constraints ----
        # daily drawdown
        if self._calculate_daily_drawdown() > MAX_DAILY_DRAWDOWN_PERCENT:
            # We can penalize heavily or end the episode
            reward -= 1.0
            done = True
        
        # overall drawdown
        if self._calculate_overall_drawdown() > MAX_OVERALL_DRAWDOWN_PERCENT:
            reward -= 2.0
            done = True
        
        obs = self._get_observation()
        info = {}
        
        return obs, reward, done, info
    
    def _update_equity(self, current_price):
        """
        Update equity based on current position and price.
        """
        if self.position_size == 0:
            self.equity = self.current_balance
        else:
            # Very simplified: assume 1 lot = 1 unit for demonstration
            # Real world: 1 lot XAUUSD typically = 100 ounces, so adjust accordingly
            pip_value = self.position_size * (current_price - self.entry_price)
            floating_pnl = pip_value
            self.equity = self.current_balance + floating_pnl
        
        # Update daily peak
        if self.equity > self.daily_peak_balance:
            self.daily_peak_balance = self.equity
        
        # Update overall peak
        if self.equity > self.overall_peak_balance:
            self.overall_peak_balance = self.equity
        
        # Update daily drawdown
        current_dd = self._calculate_daily_drawdown()
        self.daily_drawdown_percent = current_dd
    
    def _open_position(self, lots, direction, price):
        """
        Open a position: set position_size, entry_price.
        For simplicity, 1 lot = 1 unit exposure.
        """
        if direction == "long":
            self.position_size = lots
        else:
            self.position_size = -lots
        self.entry_price = price
    
    def _close_partial(self, current_price):
        """
        Close 50% of the position and realize PnL on that portion.
        """
        realized_pnl = 0
        closed_size = self.position_size * PARTIAL_CLOSE_RATIO
        remaining_size = self.position_size - closed_size
        
        # Calculate realized PnL for the closed portion
        pip_value = (current_price - self.entry_price)
        # if short position, invert the sign
        if self.position_size < 0:
            pip_value = (self.entry_price - current_price)
        
        realized_pnl = pip_value * closed_size
        self.current_balance += realized_pnl
        self.position_size = remaining_size
        
        # If the position is effectively very small, set it to zero
        if abs(self.position_size) < 1e-6:
            self.position_size = 0.0
        
        self.partial_close_triggered = True
    
    def _close_full(self, current_price):
        """
        Close full position, realize all PnL.
        """
        if self.position_size == 0:
            return
        
        pip_value = (current_price - self.entry_price)
        if self.position_size < 0:
            pip_value = (self.entry_price - current_price)
        
        realized_pnl = pip_value * abs(self.position_size)
        self.current_balance += realized_pnl
        
        # Reset position
        self.position_size = 0.0
        self.entry_price = 0.0
        self.partial_close_triggered = False
    
    def _calculate_daily_drawdown(self):
        """
        Compute current daily drawdown in % from daily peak balance.
        """
        if self.daily_peak_balance == 0:
            return 0.0
        dd = (self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0
        return max(dd, 0.0)
    
    def _calculate_overall_drawdown(self):
        """
        Compute overall drawdown in % from all-time peak balance.
        """
        if self.overall_peak_balance == 0:
            return 0.0
        dd = (self.overall_peak_balance - self.equity) / self.overall_peak_balance * 100.0
        return max(dd, 0.0)
    
    def render(self, mode='human'):
        # Optional: print out or plot info
        pass

# -------------------- MAIN SCRIPT --------------------
def main():
    # 1) Initialize MT5
    if not initialize_mt5():
        print("Failed to initialize MT5.")
        return
    
    # 2) Download historical data
    #    Example: last 180 days, 5-minute timeframe
    df = download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=180)
    if df.empty:
        print("No data, exiting.")
        return
    
    # Let's split data into train/test
    # Example: first 80% for training, remaining 20% for testing
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    
    # 3) Create training environment
    train_env = GoldTradingEnv(df_train)
    # SB3 requires a vectorized environment
    vec_train_env = DummyVecEnv([lambda: train_env])
    
    # 4) Train RL model (PPO) as an example
    model = PPO("MlpPolicy", vec_train_env, verbose=1, tensorboard_log="./ppo_xauusd_tensorboard/")
    model.learn(total_timesteps=10_000)  # Increase timesteps for real training
    
    # 5) Evaluate on test set
    test_env = GoldTradingEnv(df_test)
    obs = test_env.reset()
    
    done = False
    total_reward = 0.0
    
    while not done:
        # model.predict returns (action, states), we only need action here
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
    
    print(f"Test completed. Total reward: {total_reward:.2f}")
    print(f"Final balance: {test_env.current_balance:.2f}")
    print(f"Max daily drawdown in test: {test_env.daily_drawdown_percent:.2f}%")
    
    # 6) (Optional) Save the model
    model.save("ppo_gold_trader")
    
    # 7) Shut down MT5
    mt5.shutdown()

if __name__ == "__main__":
    main()
