import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time

# Required libraries for RL, MT5, Gym, and XGBoost
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
except ImportError:
    print("Please install torch (pip install torch).")
    sys.exit(1)

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Please install stable-baselines3 (pip install stable-baselines3).")
    sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("Please install sb3-contrib (pip install sb3-contrib).")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("Please install xgboost (pip install xgboost).")
    sys.exit(1)

##############################################
# GLOBAL CONFIGURATION
##############################################
SYMBOL = "XAUUSDm"              # Symbol to trade
INITIAL_BALANCE = 100_000.0
MAX_DAILY_DRAWDOWN_PERCENT = 4.0
MAX_OVERALL_DRAWDOWN_PERCENT = 10.0
PARTIAL_CLOSE_RATIO = 0.5       # Exit 50% of position at 1:1 profit
TRAIN_TIMEFRAMES_DAYS = 365     # 1 year of historical data

ITER_TIMESTEPS = 50_000         # Timesteps per training iteration
MAX_ITERATIONS = 20             # Maximum adaptive iterations

# Strategy parameters
BARS_N = 5                    # Lookback for breakout calculation
SH_INPUT = 0                  # Start Hour (0 means inactive)
EH_INPUT = 0                  # End Hour (0 means inactive)
TP_POINTS = 200               # (Not directly used: stops are ATR based)
SL_POINTS = 200               # (Not directly used)
TSL_TRIGGER_POINTS = 15       # Trailing stop trigger threshold (in points)
TSL_POINTS = 10               # Trailing stop shift in points
_POINT = 0.01                 # Point value

# RL risk parameters
RISK_FACTOR = 0.07            # Fraction of balance risked per trade
REWARD_MULTIPLIER = 10.0      # Base reward scaling factor
MAX_LOT_SIZE = 10.0           # Maximum allowed lot size

# Weekly/daily shaping parameters
TARGET_WEEKLY_RETURN = 0.045  # Target weekly return (4.5%)
TARGET_WEEKLY_LOWER = 0.04    
TARGET_WEEKLY_UPPER = 0.05    
WEEKLY_COEFF = 200.0          
WEEKLY_BONUS = 2000.0         
WEEKLY_PENALTY_MULTIPLIER = 30000.0

# Drawdown penalties
DAILY_DRAWDOWN_PENALTY = 1.0
OVERALL_DRAWDOWN_PENALTY = 2.0

# Trade closure reward adjustments
TRADE_BONUS = 30.0            
TRADE_PENALTY = 10.0          
TRADE_FREQUENCY_BONUS = 1.0

# Direct profit reward weight
PROFIT_REWARD_WEIGHT = 3.0

# Scalping parameter: if in range, use a reduced ATR multiplier
SCALP_ATR_MULTIPLIER = 0.5

# Extra volatility reward weight
VOLATILITY_REWARD_WEIGHT = 5.0

##############################################
# TECHNICAL INDICATOR FUNCTIONS
##############################################
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - 100/(1+rs)
    return rsi.fillna(50)

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.bfill()

def dynamic_lot_size(current_balance, atr_value, risk_factor=RISK_FACTOR, factor=100):
    if atr_value < 1e-9:
        atr_value = 1.0
    risk_amount = risk_factor * current_balance
    lots = risk_amount / (atr_value * factor)
    return max(1.0, min(lots, MAX_LOT_SIZE))

##############################################
# XGBOOST MODEL TRAINING FUNCTION
##############################################
def train_xgb_model(df):
    df_train = df.copy().reset_index(drop=True)
    df_train = df_train.copy()
    df_train['target'] = df_train['close'].shift(-1) - df_train['close']
    df_train = df_train.dropna().reset_index(drop=True)
    features = df_train[['open', 'high', 'low', 'close', 'volume']].copy()
    features['ema50'] = compute_ema(df_train['close'], span=50)
    features['atr'] = compute_atr(df_train, period=14)
    target = df_train['target']
    feature_names = ["open", "high", "low", "close", "volume", "ema50", "atr"]
    dtrain = xgb.DMatrix(features, label=target, feature_names=feature_names)
    tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"
    params = {
        "objective": "reg:squarederror",
        "tree_method": tree_method,
        "eval_metric": "rmse",
        "max_depth": 5,
        "eta": 0.1,
        "verbosity": 0
    }
    num_round = 50
    model = xgb.train(params, dtrain, num_round)
    return model

##############################################
# MT5 CONNECTION & DATA COLLECTION
##############################################
def initialize_mt5(account_id=None, password=None, server=None):
    if not mt5.initialize():
        print("MT5 Initialize() failed. Error code =", mt5.last_error())
        return False
    return True

def download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS, num_bars=0):
    utc_now = datetime.now()
    if num_bars <= 0:
        utc_from = utc_now - timedelta(days=start_days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} at timeframe {timeframe}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['ema50'] = compute_ema(df['close'], span=50)
    df['ema50_slope'] = df['ema50'].diff().fillna(0.0)
    return df

##############################################
# ADVANCED CUSTOM GYM ENVIRONMENT
##############################################
class GoldTradingEnv(gym.Env):
    """
    Advanced RL Environment for Gold Trading (XAUUSDm).

    Strategy:
      • Uses EMA50 ± 0.5×ATR to determine market regime.
      • In trend-up: open long; in trend-down: open short.
      • In range: use RSI (if RSI ≤40 → long; if RSI ≥60 → short).
      • Applies dynamic ATR-based stops, trailing stops, partial exits, and extra volatility reward.
      • Optionally includes an XGBoost forecast as an extra feature.
    
    Observation:
      Returns 16 features:
         13 base features (open, high, low, close, volume, rsi, macd_hist, atr, position, balance, daily DD%, overall peak, partial_taken),
         plus week_elapsed, weekly_return, and xgb_pred.
    
    Actions (Discrete):
         0 = Hold, 1 = Open Long, 2 = Open Short, 3 = Partial Exit, 4 = Full Exit.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=INITIAL_BALANCE, verbose=False,
                 BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                 Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                 TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                 xgb_model=None):
        super(GoldTradingEnv, self).__init__()
        self.df = df.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.verbose = verbose
        self.BarsN = BarsN
        self.SHInput = SHInput
        self.EHInput = EHInput
        self.TslTriggerPoints = TslTriggerPoints
        self.TslPoints = TslPoints
        self.Tppoints = Tppoints
        self.Slpoints = Slpoints
        self.xgb_model = xgb_model
        
        self.df['rsi'] = compute_rsi(self.df['close'], period=14).fillna(50.0)
        self.df['macd_hist'] = 0.0  # Placeholder for MACD
        self.df['atr'] = compute_atr(self.df, period=14).fillna(0.0)
        if 'ema50' not in self.df.columns:
            self.df['ema50'] = compute_ema(self.df['close'], span=50).fillna(0.0)
            self.df['ema50_slope'] = self.df['ema50'].diff().fillna(0.0)
        
        self.current_step = 0
        self.position = 0         # 0: flat, 1: long, -1: short
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.partial_taken = False
        self.trade_log = []
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.daily_peak_balance = initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = initial_balance
        
        self.week_start_time = self.df.iloc[0]["time"]
        self.week_start_balance = initial_balance
        
        # Observation space is fixed to 16 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
    
    def _get_regime(self, price, slow_ma, atr):
        upper_bound = slow_ma + 0.5 * atr
        lower_bound = slow_ma - 0.5 * atr
        if price > upper_bound:
            return "trend_up"
        elif price < lower_bound:
            return "trend_down"
        else:
            return "range"
    
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_time = row["time"]
        week_elapsed = (current_time - self.week_start_time).total_seconds() / (7*24*3600)
        week_elapsed = min(week_elapsed, 1.0)
        weekly_return = (self.current_balance / self.week_start_balance) - 1
        base_obs = np.array([
            row.open, row.high, row.low, row.close, row.volume, row.rsi,
            row.macd_hist, row.atr, float(self.position), self.current_balance,
            self.daily_drawdown_percent, self.overall_peak_balance, float(self.partial_taken)
        ], dtype=np.float32)
        extra_obs = np.array([week_elapsed, weekly_return], dtype=np.float32)
        obs = np.concatenate((base_obs, extra_obs))
        if self.xgb_model is not None:
            feature_names = ["open", "high", "low", "close", "volume", "ema50", "atr"]
            features = np.array([row.open, row.high, row.low, row.close, row.volume, row.ema50, row.atr]).reshape(1, -1)
            try:
                dmatrix = xgb.DMatrix(features, feature_names=feature_names)
                xgb_pred = self.xgb_model.predict(dmatrix)[0]
            except Exception as e:
                print("XGBoost prediction error:", e)
                xgb_pred = 0.0
        else:
            xgb_pred = 0.0
        obs = np.concatenate((obs, np.array([xgb_pred], dtype=np.float32)))
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.partial_taken = False
        self.trade_log = []
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.daily_peak_balance = self.initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = self.initial_balance
        self.week_start_time = self.df.iloc[0]["time"]
        self.week_start_balance = self.initial_balance
        return self._get_observation()
    
    def _update_equity(self, price):
        if self.position == 0:
            self.equity = self.current_balance
        else:
            if self.position == 1:
                unrealized = (price - self.entry_price) * self.position_size
            else:
                unrealized = (self.entry_price - price) * self.position_size
            self.equity = self.current_balance + unrealized
        self.daily_peak_balance = max(self.daily_peak_balance, self.equity)
        self.overall_peak_balance = max(self.overall_peak_balance, self.equity)
        self.daily_drawdown_percent = max((self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0, 0.0)
    
    def _trade_closure_bonus(self, pnl):
        if pnl > 0:
            return TRADE_BONUS
        elif pnl < 0:
            return -TRADE_PENALTY
        return 0.0
    
    def _close_position(self, price):
        if self.position == 0 or self.position_size == 0:
            return 0.0
        if self.position == 1:
            pnl = (price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - price) * self.position_size
        bonus = self._trade_closure_bonus(pnl)
        self.current_balance += pnl + bonus
        # Log trade closure with balance info to allow win rate calculation
        self.trade_log.append({
            "step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": price,
            "position": "LONG" if self.position == 1 else "SHORT",
            "profit": pnl,
            "bonus": bonus,
            "action": 4,
            "old_balance": self.current_balance - pnl - bonus,
            "new_balance": self.current_balance
        })
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.partial_taken = False
        return bonus
    
    def _partial_exit(self, price):
        if self.position == 0 or self.partial_taken:
            return 0.0
        if not hasattr(self, 'initial_stop'):
            return 0.0
        risk = abs(self.entry_price - self.initial_stop)
        if risk <= 0:
            return 0.0
        bonus = 0.0
        if self.position == 1:
            profit = price - self.entry_price
            if profit >= risk:
                half = self.position_size * PARTIAL_CLOSE_RATIO
                realized = profit * half
                bonus = self._trade_closure_bonus(realized)
                self.current_balance += realized + bonus
                self.position_size -= half
                self.partial_taken = True
                if self.position_size < 1e-6:
                    self.position = 0
                    self.stop_price = 0.0
        else:
            profit = self.entry_price - price
            if profit >= risk:
                half = self.position_size * PARTIAL_CLOSE_RATIO
                realized = profit * half
                bonus = self._trade_closure_bonus(realized)
                self.current_balance += realized + bonus
                self.position_size -= half
                self.partial_taken = True
                if self.position_size < 1e-6:
                    self.position = 0
                    self.stop_price = 0.0
        return bonus
    
    def _update_trailing_stop(self, price):
        if self.position == 0 or not hasattr(self, 'initial_stop'):
            return
        risk = abs(self.entry_price - self.initial_stop)
        if risk <= 0:
            return
        if self.position == 1:
            profit = price - self.entry_price
            if profit >= 2 * risk and self.stop_price < self.entry_price + risk:
                self.stop_price = self.entry_price + risk
            if profit >= 3 * risk and self.stop_price < self.entry_price + 2 * risk:
                self.stop_price = self.entry_price + 2 * risk
        else:
            profit = self.entry_price - price
            if profit >= 2 * risk and self.stop_price > self.entry_price - risk:
                self.stop_price = self.entry_price - risk
            if profit >= 3 * risk and self.stop_price > self.entry_price - 2 * risk:
                self.stop_price = self.entry_price - 2 * risk
    
    def _check_stop_loss_hit(self, price):
        if self.position == 1 and price <= self.stop_price:
            return True
        if self.position == -1 and price >= self.stop_price:
            return True
        return False
    
    def _get_entry_signal(self, price):
        row = self.df.iloc[self.current_step]
        slow_ma = row.ema50
        atr_val = row.atr
        rsi_val = row.rsi
        regime = self._get_regime(price, slow_ma, atr_val)
        if regime == "trend_up":
            return 1, atr_val
        elif regime == "trend_down":
            return -1, atr_val
        else:
            if rsi_val <= 40:
                return 1, SCALP_ATR_MULTIPLIER * atr_val
            elif rsi_val >= 60:
                return -1, SCALP_ATR_MULTIPLIER * atr_val
            else:
                return 0, 0
    
    def _execute_action(self, action):
        price = self.df.iloc[self.current_step].close
        current_time = self.df.iloc[self.current_step]["time"]
        if self.SHInput and current_time.hour < self.SHInput:
            if self.position != 0:
                self._close_position(price)
            return
        if self.EHInput and self.EHInput != 0 and current_time.hour > self.EHInput:
            if self.position != 0:
                self._close_position(price)
            return
        
        if action == 0:
            self.closure_bonus = TRADE_FREQUENCY_BONUS
        elif action == 1:
            if self.position == 0:
                signal, stop_dist = self._get_entry_signal(price)
                if signal == 1:
                    self.position = 1
                    self.entry_price = price
                    self.initial_stop = price - stop_dist
                    self.position_size = dynamic_lot_size(self.current_balance, stop_dist)
                    self.stop_price = self.initial_stop
                    self.partial_taken = False
        elif action == 2:
            if self.position == 0:
                signal, stop_dist = self._get_entry_signal(price)
                if signal == -1:
                    self.position = -1
                    self.entry_price = price
                    self.initial_stop = price + stop_dist
                    self.position_size = dynamic_lot_size(self.current_balance, stop_dist)
                    self.stop_price = self.initial_stop
                    self.partial_taken = False
        elif action == 3:
            bonus = self._partial_exit(price)
            self.closure_bonus = bonus
        elif action == 4:
            bonus = self._close_position(price)
            self.closure_bonus = bonus
    
    def step(self, action):
        price = self.df.iloc[self.current_step].close
        old_equity = self.equity
        old_balance = self.current_balance
        old_position = self.position
        
        self._update_equity(price)
        if self.position != 0:
            self._update_trailing_stop(price)
            if self._check_stop_loss_hit(price):
                self.closure_bonus = self._close_position(price)
        self._execute_action(action)
        if self.position != 0:
            self._partial_exit(price)
        
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        if not done:
            nxt_price = self.df.iloc[self.current_step].close
            self._update_equity(nxt_price)
        else:
            self._update_equity(price)
        
        reward = ((self.equity - old_equity) / self.initial_balance) * REWARD_MULTIPLIER
        profit_reward = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100.0 * PROFIT_REWARD_WEIGHT
        reward += profit_reward
        if hasattr(self, 'closure_bonus'):
            reward += self.closure_bonus
            self.closure_bonus = 0.0
        
        row = self.df.iloc[self.current_step]
        current_time = row["time"]
        week_elapsed = (current_time - self.week_start_time).total_seconds() / (7*24*3600)
        week_elapsed = min(week_elapsed, 1.0)
        weekly_return = (self.current_balance / self.week_start_balance) - 1
        continuous_reward = -WEEKLY_COEFF * (weekly_return - TARGET_WEEKLY_RETURN) ** 2 * week_elapsed
        reward += continuous_reward
        
        if (current_time.isocalendar().year != self.week_start_time.isocalendar().year or
            current_time.isocalendar().week != self.week_start_time.isocalendar().week):
            if TARGET_WEEKLY_LOWER <= weekly_return <= TARGET_WEEKLY_UPPER:
                reward += WEEKLY_BONUS
            else:
                reward -= WEEKLY_PENALTY_MULTIPLIER * abs(weekly_return - TARGET_WEEKLY_RETURN)
            self.week_start_time = current_time
            self.week_start_balance = self.current_balance
        
        if self._calculate_daily_drawdown() > MAX_DAILY_DRAWDOWN_PERCENT:
            reward -= DAILY_DRAWDOWN_PENALTY
            done = True
        if self._calculate_overall_drawdown() > MAX_OVERALL_DRAWDOWN_PERCENT:
            reward -= OVERALL_DRAWDOWN_PENALTY
            done = True
        
        atr_val = row.atr
        if atr_val > 0:
            risk_adj = (self.current_balance - old_balance) / (atr_val + 1e-9)
            reward += VOLATILITY_REWARD_WEIGHT * risk_adj
        
        step_log = {
            "step": self.current_step,
            "action": action,
            "old_position": old_position,
            "new_position": self.position,
            "price": price,
            "old_equity": old_equity,
            "new_equity": self.equity,
            "old_balance": old_balance,
            "new_balance": self.current_balance,
            "reward": reward,
            "daily_dd%": self._calculate_daily_drawdown(),
            "overall_dd%": self._calculate_overall_drawdown()
        }
        self.trade_log.append(step_log)
        if self.verbose:
            print(f"[Step {self.current_step}] Action={action}, Price={price:.2f}, Equity={self.equity:.2f}, Reward={reward:.4f}, Position={self.position}")
        return self._get_observation(), reward, done, {}
    
    def _calculate_daily_drawdown(self):
        if self.daily_peak_balance <= 0:
            return 0.0
        return max((self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0, 0.0)
    
    def _calculate_overall_drawdown(self):
        if self.overall_peak_balance <= 0:
            return 0.0
        return max((self.overall_peak_balance - self.equity) / self.overall_peak_balance * 100.0, 0.0)
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.current_balance:.2f}, Equity: {self.equity:.2f}")
    
    def update_risk_factor(self, new_risk):
        self.risk_factor = new_risk
        self.max_position = max(1, int(self.risk_factor * 10))
        self.observation_space.high[8] = float(self.max_position)
        self.observation_space.low[8] = -float(self.max_position)
    
    def get_trade_log(self):
        return self.trade_log

##############################################
# UTILITY: Compute Win Rate
##############################################
def compute_win_rate(trade_log):
    # Only include log entries that contain 'new_balance'
    closure_trades = [entry for entry in trade_log if entry.get("action") in [3, 4] and "new_balance" in entry]
    if not closure_trades:
        return None
    wins = sum(1 for entry in closure_trades if entry["new_balance"] > entry["old_balance"])
    return wins / len(closure_trades)

##############################################
# BACKTEST FUNCTION FOR LAST WEEK
##############################################
def backtest_last_week(model, df_test, initial_balance=INITIAL_BALANCE):
    last_week = df_test['time'].apply(lambda t: t.isocalendar().week).max()
    df_last_week = df_test[df_test['time'].apply(lambda t: t.isocalendar().week) == last_week].copy()
    if df_last_week.empty:
        print("No data for the last week.")
        return None
    print(f"Backtesting on last week (ISO week {last_week})...")
    env_last_week = GoldTradingEnv(df_last_week, initial_balance=initial_balance, verbose=True,
                                   BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                                   Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                                   TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                                   xgb_model=model.env.envs[0].xgb_model)
    obs = env_last_week.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env_last_week.step(action)
        total_reward += reward
    print("Backtest results for last week:")
    print(f"Final balance: {env_last_week.current_balance:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Max daily drawdown: {env_last_week.daily_drawdown_percent:.2f}%")
    win_rate = compute_win_rate(env_last_week.trade_log)
    if win_rate is not None:
        print(f"Win rate: {win_rate * 100:.2f}%")
    else:
        print("No trade closures to compute win rate.")
    return {
        "final_balance": env_last_week.current_balance,
        "total_reward": total_reward,
        "max_daily_drawdown": env_last_week.daily_drawdown_percent,
        "win_rate": win_rate
    }

##############################################
# MAIN SCRIPT WITH ADAPTIVE TRAINING LOOP
##############################################
def main():
    log_filename = "terminal_log.txt"
    sys.stdout = open(log_filename, "w")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training using device: {device}")
    print("Torch CUDA availability:", torch.cuda.is_available())
    
    if not initialize_mt5():
        print("Failed to initialize MT5.")
        return
    
    df = download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS)
    if df.empty:
        print("No data retrieved, exiting.")
        return
    
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    
    print("Training XGBoost forecasting model...")
    xgb_model = train_xgb_model(df_train)
    print("XGBoost model trained.")
    
    def make_train_env():
        return GoldTradingEnv(df_train, initial_balance=INITIAL_BALANCE, verbose=False,
                              BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                              Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                              TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                              xgb_model=xgb_model)
    vec_train_env = DummyVecEnv([make_train_env])
    
    # Note: Removed lstm-specific parameters to avoid unexpected keyword errors.
    policy_kwargs = dict(net_arch=[256, 256, 256])
    model = RecurrentPPO("MlpLstmPolicy", vec_train_env, verbose=1,
                         tensorboard_log="./ppo_xauusdm_tensorboard/",
                         learning_rate=1e-4, gamma=0.99,
                         policy_kwargs=policy_kwargs, device=device)
    
    target_weekly_profit = TARGET_WEEKLY_RETURN  # 4.5%
    iteration = 0
    achieved = False
    while iteration < MAX_ITERATIONS:
        print(f"\nAdaptive Training Iteration {iteration}")
        print(f"Training for {ITER_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=ITER_TIMESTEPS, reset_num_timesteps=False)
        backtest_results = backtest_last_week(model, df_test, initial_balance=INITIAL_BALANCE)
        if backtest_results is None:
            break
        weekly_profit = (backtest_results["final_balance"] / INITIAL_BALANCE) - 1
        win_rate = backtest_results.get("win_rate")
        print(f"Iteration {iteration} -- Weekly profit: {weekly_profit:.2%}")
        if weekly_profit >= target_weekly_profit and (win_rate is not None and win_rate >= 0.90):
            print("Target weekly profit and win rate achieved!")
            achieved = True
            break
        else:
            global RISK_FACTOR, WEEKLY_BONUS
            RISK_FACTOR = min(RISK_FACTOR + 0.01, 0.1)
            WEEKLY_BONUS *= 1.2
            print(f"Adaptive update: New RISK_FACTOR = {RISK_FACTOR:.2f}, New WEEKLY_BONUS = {WEEKLY_BONUS:.2f}")
        iteration += 1
        vec_train_env = DummyVecEnv([make_train_env])
        model.set_env(vec_train_env)
        time.sleep(1)
    
    if not achieved:
        print("Maximum adaptive iterations reached without meeting target weekly profit and win rate.")
    
    test_env = GoldTradingEnv(df_test, initial_balance=INITIAL_BALANCE, verbose=True,
                              BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                              Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                              TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                              xgb_model=xgb_model)
    obs = test_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = test_env.step(action)
        total_reward += reward
    print(f"\nFinal Test on Full Test Set:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final balance on test set: {test_env.current_balance:.2f}")
    print(f"Max daily drawdown on test set: {test_env.daily_drawdown_percent:.2f}%")
    win_rate = compute_win_rate(test_env.trade_log)
    if win_rate is not None:
        print(f"Win rate on test set: {win_rate * 100:.2f}%")
    else:
        print("No trade closures to compute win rate.")
    trade_log_df = pd.DataFrame(test_env.trade_log)
    trade_log_df.to_csv("test_trade_log.csv", index=False)
    print("Saved full test trade log to test_trade_log.csv")
    
    final_backtest = backtest_last_week(model, df_test, initial_balance=INITIAL_BALANCE)
    if final_backtest is not None:
        print("Final backtest (last week) summary:", final_backtest)
    
    model.save("advanced_ppo_gold_trader")
    mt5.shutdown()
    sys.stdout.close()

if __name__ == "__main__":
    main()
