import os
import sys
import time
from datetime import datetime, timedelta, time as dt_time, timezone
import numpy as np
import pandas as pd

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
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)
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
SYMBOL = "XAUUSDm"                  # Symbol to trade
INITIAL_BALANCE = 100_000.0
MAX_DAILY_DRAWDOWN_PERCENT = 4.0    # Maximum daily drawdown (4%)
MAX_OVERALL_DRAWDOWN_PERCENT = 10.0
PARTIAL_CLOSE_RATIO = 0.5           # Exit 50% of position at 1:1 profit
TRAIN_TIMEFRAMES_DAYS = 365         # 1 year of historical data

# Increase timesteps per iteration to allow longer training per adaptive cycle
ITER_TIMESTEPS = 100_000            

# New York Session Parameters (UTC)
NY_START_HOUR = 13
NY_START_MINUTE = 30
NY_END_HOUR = 20
NY_END_MINUTE = 0

SH_INPUT = NY_START_HOUR          # Session start hour
EH_INPUT = NY_END_HOUR            # Session end hour

# Strategy parameters
BARS_N = 5                      # Lookback for breakout calculation
TP_POINTS = 200                 # (Not directly used: stops are ATR based)
SL_POINTS = 200                 # (Not directly used)
TSL_TRIGGER_POINTS = 15         # Trailing stop trigger threshold (in points)
TSL_POINTS = 10                 # Trailing stop shift in points
_POINT = 0.01                   # Point value

# RL risk parameters
RISK_FACTOR = 0.02              # Starting risk factor (will update adaptively)
REWARD_MULTIPLIER = 20.0        # Base reward multiplier (used in reward calc)
# Increased profit reward weight to further emphasize winning trades.
PROFIT_REWARD_WEIGHT = 20.0     

# Maximum allowed lot size
MAX_LOT_SIZE = 10.0             

# Weekly/daily shaping parameters
TARGET_WEEKLY_RETURN = 0.045    # Target weekly return (4.5%)
TARGET_WEEKLY_LOWER = 0.04      
TARGET_WEEKLY_UPPER = 0.05      
WEEKLY_COEFF = 200.0            
WEEKLY_BONUS = 5000.0           # Increased weekly bonus
WEEKLY_PENALTY_MULTIPLIER = 15000.0  # Reduced penalty multiplier

# Drawdown penalties
DAILY_DRAWDOWN_PENALTY = 1.0
OVERALL_DRAWDOWN_PENALTY = 2.0

# Trade closure reward adjustments
TRADE_BONUS = 50.0            # Increased bonus for profitable trade closures
TRADE_PENALTY = 10.0          
TRADE_FREQUENCY_BONUS = 1.0

# Scalping parameter: use reduced ATR multiplier
SCALP_ATR_MULTIPLIER = 0.5

# Extra volatility reward weight
VOLATILITY_REWARD_WEIGHT = 5.0

# In dynamic lot sizing, reduce factor from 100 to 50 to allow larger lots.
LOT_SIZE_FACTOR = 50

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

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_hist.fillna(0.0)

def dynamic_lot_size(current_balance, atr_value, confidence=1.0, risk_factor=RISK_FACTOR, factor=LOT_SIZE_FACTOR):
    if atr_value < 1e-9:
        atr_value = 1.0
    risk_amount = risk_factor * current_balance * confidence
    lots = risk_amount / (atr_value * factor)
    return max(0.01, min(lots, MAX_LOT_SIZE))

##############################################
# XGBOOST MODEL TRAINING FUNCTION
##############################################
def train_xgb_model(df):
    df_train = df.copy().reset_index(drop=True)
    df_train['target'] = df_train['close'].shift(-1) - df_train['close']
    df_train = df_train.dropna().reset_index(drop=True)
    features = df_train[['open', 'high', 'low', 'close', 'volume']].copy()
    features['ema50'] = compute_ema(df_train['close'], span=50)
    features['atr'] = compute_atr(df_train, period=14)
    target = df_train['target']
    feature_names = ["open", "high", "low", "close", "volume", "ema50", "atr"]
    dtrain = xgb.DMatrix(features, label=target, feature_names=feature_names)
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
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
    utc_now = datetime.now(timezone.utc)
    if num_bars <= 0:
        utc_from = utc_now - timedelta(days=start_days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} at timeframe {timeframe}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['ema50'] = compute_ema(df['close'], span=50)
    df['ema50_slope'] = df['ema50'].diff().fillna(0.0)
    df['rsi'] = compute_rsi(df['close'], period=14)
    df['atr'] = compute_atr(df, period=14)
    df['macd_hist'] = compute_macd(df['close'])
    def is_new_york_session(ts):
        session_start = dt_time(NY_START_HOUR, NY_START_MINUTE)
        session_end = dt_time(NY_END_HOUR, NY_END_MINUTE)
        return session_start <= ts.time() <= session_end
    df = df[df['time'].apply(is_new_york_session)].reset_index(drop=True)
    return df

##############################################
# ADVANCED CUSTOM GYM ENVIRONMENT
##############################################
class GoldTradingEnv(gym.Env):
    """
    Advanced RL Environment for Gold Trading (XAUUSDm).

    Observations: 16 features:
      [open, high, low, close, volume, rsi, macd_hist, atr,
       position, balance_ratio, daily_drawdown, peak_balance_ratio,
       partial_exit_flag, week_elapsed, weekly_return, xgb_pred]

    Actions:
      0 = Hold, 1 = Open Long, 2 = Open Short, 3 = Partial Exit, 4 = Full Exit.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, df, initial_balance=INITIAL_BALANCE, verbose=False,
                 BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                 Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                 TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                 xgb_model=None, render_mode='human'):
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
        self.Slpoints = SL_POINTS
        self.xgb_model = xgb_model

        self.df['rsi'] = compute_rsi(self.df['close'], period=14).fillna(50.0)
        self.df['macd_hist'] = compute_macd(self.df['close']).fillna(0.0)
        self.df['atr'] = compute_atr(self.df, period=14).fillna(0.0)
        if 'ema50' not in self.df.columns:
            self.df['ema50'] = compute_ema(self.df['close'], span=50).fillna(0.0)
        self.df['ema50_slope'] = self.df['ema50'].diff().fillna(0.0)

        self.current_step = 0
        self.position = 0           # 0: flat, 1: long, -1: short
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
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.render_mode = render_mode
        self.closure_bonus = 0.0

        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def _scale_val(self, val, old_min, old_max, new_min=-1.0, new_max=1.0):
        if abs(old_max - old_min) < 1e-9:
            return 0.0
        return new_min + (val - old_min) * (new_max - new_min) / (old_max - old_min)

    def _get_confidence(self):
        row = self.df.iloc[self.current_step]
        feature_names = ["open", "high", "low", "close", "volume", "ema50", "atr"]
        features = np.array([row.open, row.high, row.low, row.close,
                             row.volume, row.ema50, row.atr]).reshape(1, -1)
        try:
            dmatrix = xgb.DMatrix(features, feature_names=feature_names)
            xgb_pred = self.xgb_model.predict(dmatrix)[0]
            confidence = min(abs(xgb_pred) / 10.0, 1.0)
            return confidence
        except:
            return 0.5

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_time = row["time"]
        week_elapsed = (current_time - self.week_start_time).total_seconds() / (7 * 24 * 3600)
        week_elapsed = min(week_elapsed, 1.0)
        weekly_return = (self.current_balance / self.week_start_balance) - 1

        open_s = self._scale_val(row.open, self.df['open'].min(), self.df['open'].max())
        high_s = self._scale_val(row.high, self.df['high'].min(), self.df['high'].max())
        low_s  = self._scale_val(row.low,  self.df['low'].min(),  self.df['low'].max())
        close_s= self._scale_val(row.close,self.df['close'].min(),self.df['close'].max())
        vol_s  = self._scale_val(row.volume,self.df['volume'].min(),self.df['volume'].max())
        rsi_s  = self._scale_val(row.rsi,   0.0, 100.0)
        macd_s = self._scale_val(row.macd_hist, -5.0, 5.0)
        atr_s  = self._scale_val(row.atr, 0.0, self.df['atr'].max())

        balance_ratio = self._scale_val(self.current_balance / self.initial_balance, 0.0, 10.0)
        daily_dd = self._scale_val(self.daily_drawdown_percent, 0.0, 50.0)
        peak_balance = self._scale_val(self.overall_peak_balance / self.initial_balance, 1.0, 10.0)

        xgb_pred = 0.0
        if self.xgb_model is not None:
            try:
                feature_names = ["open", "high", "low", "close", "volume", "ema50", "atr"]
                features = np.array([row.open, row.high, row.low, row.close,
                                     row.volume, row.ema50, row.atr]).reshape(1, -1)
                dmatrix = xgb.DMatrix(features, feature_names=feature_names)
                xgb_pred = float(self.xgb_model.predict(dmatrix)[0])
            except:
                xgb_pred = 0.0

        obs = np.array([
            open_s, high_s, low_s, close_s, vol_s, rsi_s,
            macd_s, atr_s,
            float(self.position),
            balance_ratio,
            daily_dd,
            peak_balance,
            float(self.partial_taken),
            week_elapsed,
            weekly_return,
            xgb_pred
        ], dtype=np.float32)
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
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.closure_bonus = 0.0
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
        pnl = (price - self.entry_price) * self.position_size if self.position == 1 else (self.entry_price - price) * self.position_size
        bonus = self._trade_closure_bonus(pnl)
        self.current_balance += pnl + bonus
        # Extra bonus if trade profit exceeds threshold (e.g., 0.1% gain)
        if pnl > 0 and pnl > 0.001 * self.current_balance:
            bonus += 50  # extra bonus
            self.current_balance += 50
        trade_result = {
            "step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": price,
            "position": "LONG" if self.position == 1 else "SHORT",
            "profit": pnl,
            "bonus": bonus,
            "action": 4,
            "old_balance": self.current_balance - pnl - bonus,
            "new_balance": self.current_balance,
            "timestamp": datetime.now().isoformat()
        }
        self.trade_log.append(trade_result)
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.partial_taken = False
        self.total_trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        if self.verbose:
            print(f"Trade {self.total_trades}: {'WIN' if pnl > 0 else 'LOSS'} | Profit: {pnl:.2f} | Balance: {self.current_balance:.2f}")
        return bonus

    def _partial_exit(self, price):
        if self.position == 0 or self.partial_taken or not hasattr(self, 'initial_stop'):
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

    # UPDATED: Tighter entry signal thresholds to filter out low-confidence trades.
    def _get_entry_signal(self, price):
        row = self.df.iloc[self.current_step]
        slow_ma = row.ema50
        atr_val = row.atr
        rsi_val = row.rsi
        macd_val = row.macd_hist
        regime = self._get_regime(price, slow_ma, atr_val)
        if regime == "trend_up":
            # Only enter long if RSI is moderately low to avoid overbought conditions.
            if rsi_val < 45:
                return 1, atr_val
            else:
                return 0, 0
        elif regime == "trend_down":
            # Only enter short if RSI is moderately high.
            if rsi_val > 55:
                return -1, atr_val
            else:
                return 0, 0
        else:
            # For range-bound, require even stronger signals.
            if rsi_val <= 35 and macd_val > 0.5:
                return 1, SCALP_ATR_MULTIPLIER * atr_val
            elif rsi_val >= 65 and macd_val < -0.5:
                return -1, SCALP_ATR_MULTIPLIER * atr_val
            else:
                return 0, 0

    def _get_regime(self, price, slow_ma, atr):
        upper_bound = slow_ma + 0.5 * atr
        lower_bound = slow_ma - 0.5 * atr
        if price > upper_bound:
            return "trend_up"
        elif price < lower_bound:
            return "trend_down"
        else:
            return "range"

    def _execute_action(self, action):
        price = self.df.iloc[self.current_step].close
        current_time = self.df.iloc[self.current_step]["time"]
        session_start = current_time.replace(hour=NY_START_HOUR, minute=NY_START_MINUTE, second=0, microsecond=0)
        session_end = current_time.replace(hour=NY_END_HOUR, minute=NY_END_MINUTE, second=0, microsecond=0)
        if current_time < session_start or current_time > session_end:
            if self.position != 0:
                self._close_position(price)
            return
        if action == 0:
            self.closure_bonus = TRADE_FREQUENCY_BONUS
        elif action == 1:
            if self.position == 0:
                signal, atr_val = self._get_entry_signal(price)
                confidence = self._get_confidence()
                if signal == 1:
                    self.position = 1
                    self.entry_price = price
                    self.initial_stop = price - atr_val
                    stop_dist = abs(price - self.initial_stop)
                    self.position_size = dynamic_lot_size(self.current_balance, stop_dist, confidence, risk_factor=RISK_FACTOR)
                    self.stop_price = self.initial_stop
                    self.partial_taken = False
                    if self.verbose:
                        print(f"LONG at {price:.2f}, size={self.position_size:.2f}, stop={self.stop_price:.2f}")
                else:
                    self.closure_bonus = -TRADE_PENALTY
        elif action == 2:
            if self.position == 0:
                signal, atr_val = self._get_entry_signal(price)
                confidence = self._get_confidence()
                if signal == -1:
                    self.position = -1
                    self.entry_price = price
                    self.initial_stop = price + atr_val
                    stop_dist = abs(price - self.initial_stop)
                    self.position_size = dynamic_lot_size(self.current_balance, stop_dist, confidence, risk_factor=RISK_FACTOR)
                    self.stop_price = self.initial_stop
                    self.partial_taken = False
                    if self.verbose:
                        print(f"SHORT at {price:.2f}, size={self.position_size:.2f}, stop={self.stop_price:.2f}")
                else:
                    self.closure_bonus = -TRADE_PENALTY
        elif action == 3:
            bonus = self._partial_exit(price)
            self.closure_bonus = bonus
        elif action == 4:
            bonus = self._close_position(price)
            self.closure_bonus = bonus
        else:
            self.closure_bonus = -TRADE_PENALTY

    def step(self, action):
        self._execute_action(action)
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        price = self.df.iloc[self.current_step].close
        self._update_equity(price)
        self._update_trailing_stop(price)
        if self._check_stop_loss_hit(price):
            self._close_position(price)
        reward = self._calculate_reward()
        obs = self._get_observation()
        info = {
            "balance": self.current_balance,
            "equity": self.equity,
            "daily_drawdown": self.daily_drawdown_percent,
            "position": self.position
        }
        return obs, reward, done, info

    # UPDATED: Adjusted reward function to put greater emphasis on profitable trades 
    # and meeting the weekly profit target of 4-5%.
    def _calculate_reward(self):
        current_time = self.df.iloc[self.current_step]["time"]
        weekly_elapsed = (current_time - self.week_start_time).total_seconds() / (7*24*3600)
        weekly_elapsed = min(weekly_elapsed, 1.0)
        weekly_return = (self.current_balance / self.week_start_balance) - 1
        weekly_target_weight = WEEKLY_COEFF * weekly_elapsed
        if TARGET_WEEKLY_LOWER <= weekly_return <= TARGET_WEEKLY_UPPER:
            weekly_reward = WEEKLY_BONUS
        else:
            weekly_reward = -WEEKLY_PENALTY_MULTIPLIER * abs(weekly_return - TARGET_WEEKLY_RETURN)
        daily_drawdown_penalty = -DAILY_DRAWDOWN_PENALTY * self.daily_drawdown_percent
        overall_drawdown_penalty = -OVERALL_DRAWDOWN_PENALTY * max(0.0, (self.initial_balance - self.overall_peak_balance) / self.initial_balance * 100.0)
        if self.position != 0:
            if self.position == 1:
                current_trade_pnl = (self.df.iloc[self.current_step].close - self.entry_price) * self.position_size
            else:
                current_trade_pnl = (self.entry_price - self.df.iloc[self.current_step].close) * self.position_size
            profit_reward = PROFIT_REWARD_WEIGHT * current_trade_pnl
        else:
            profit_reward = 0.0
        # Extra bonus if profit exceeds 0.1% of current balance
        extra_bonus = 0.0
        if self.position != 0 and profit_reward > 0 and (self.df.iloc[self.current_step].close - self.entry_price) * self.position_size > 0.001 * self.current_balance:
            extra_bonus = 50
        atr = self.df.iloc[self.current_step].atr
        volatility_reward = VOLATILITY_REWARD_WEIGHT * atr
        total_reward = (REWARD_MULTIPLIER * (profit_reward + volatility_reward)
                        + weekly_reward * weekly_target_weight
                        + daily_drawdown_penalty
                        + overall_drawdown_penalty
                        + extra_bonus)
        total_reward += self.closure_bonus
        self.closure_bonus = 0.0
        return total_reward

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: {self.current_balance:.2f}, Equity: {self.equity:.2f}, Daily DD: {self.daily_drawdown_percent:.2f}%, Position: {self.position}, Size: {self.position_size:.2f}")
        elif mode == 'rgb_array':
            return None

##############################################
# UTILITY: Compute Win Rate
##############################################
def compute_win_rate(trade_log):
    closure_trades = [entry for entry in trade_log if entry.get("action") in [3, 4] and "new_balance" in entry]
    if not closure_trades:
        return None
    wins = sum(1 for entry in closure_trades if entry["new_balance"] > entry["old_balance"])
    return wins / len(closure_trades)

##############################################
# BACKTEST FUNCTION FOR MULTI-WEEK
##############################################
def backtest_multi_week(model, df_test, weeks=4, initial_balance=INITIAL_BALANCE, xgb_model=None):
    df_test = df_test.copy()
    df_test['iso_week'] = df_test['time'].apply(lambda t: (t.isocalendar().year, t.isocalendar().week))
    grouped = df_test.groupby('iso_week')
    groups = sorted(list(grouped), key=lambda x: (x[0][0], x[0][1]))
    if len(groups) < weeks:
        print("Not enough weeks for multi-week backtest.")
        return None
    selected_groups = groups[-weeks:]
    weekly_profits = []
    overall_wins = 0
    overall_closures = 0
    overall_max_dd = 0.0
    balance = initial_balance
    for key, group in selected_groups:
        env = GoldTradingEnv(group.reset_index(drop=True), initial_balance=balance, verbose=False,
                             BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                             Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                             TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                             xgb_model=xgb_model, render_mode='human')
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
        week_profit = (env.current_balance / balance) - 1
        weekly_profits.append(week_profit)
        week_win_rate = compute_win_rate(env.trade_log)
        if week_win_rate is not None:
            closures = len([e for e in env.trade_log if e.get("action") in [3, 4]])
            wins = int(week_win_rate * closures)
            overall_wins += wins
            overall_closures += closures
        overall_max_dd = max(overall_max_dd, env.daily_drawdown_percent)
        balance = env.current_balance
    average_weekly_profit = sum(weekly_profits) / len(weekly_profits)
    overall_win_rate = overall_wins / overall_closures if overall_closures > 0 else 0.0
    return {"weekly_profits": weekly_profits, "average_weekly_profit": average_weekly_profit, "win_rate": overall_win_rate, "max_daily_drawdown": overall_max_dd}

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
        sys.exit(1)
    df = download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS)
    if df.empty:
        print("No data retrieved, exiting.")
        mt5.shutdown()
        sys.exit(1)
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
                              xgb_model=xgb_model, render_mode='human')
    vec_train_env = DummyVecEnv([make_train_env])
    # Upgrade network architecture: three layers of 512 neurons each.
    policy_kwargs = dict(net_arch=[512, 512, 512])
    base_lr = 1e-4
    model = RecurrentPPO("MlpLstmPolicy", vec_train_env, verbose=1,
                         tensorboard_log="./ppo_xauusdm_tensorboard/",
                         learning_rate=base_lr, gamma=0.99,
                         policy_kwargs=policy_kwargs, device=device)
    target_win_rate = 0.90  # UPDATED: Aim for 90% win rate
    multi_week = 4          # Backtest over 4 weeks
    best_metrics = None
    best_iteration = -1
    best_model = None
    iteration = 0
    # Adaptive training loop with more aggressive updates.
    while True:
        print(f"\nAdaptive Training Iteration {iteration}")
        print(f"Training for {ITER_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=ITER_TIMESTEPS, reset_num_timesteps=False)
        backtest_results = backtest_multi_week(model, df_test, weeks=multi_week, initial_balance=INITIAL_BALANCE, xgb_model=xgb_model)
        if backtest_results is None:
            break
        weekly_profits = backtest_results["weekly_profits"]
        avg_week_profit = backtest_results["average_weekly_profit"]
        win_rate = backtest_results["win_rate"]
        max_dd = backtest_results["max_daily_drawdown"]
        print(f"Iteration {iteration} -- Avg Weekly Profit: {avg_week_profit:.2%}, Win rate: {win_rate*100:.2f}%, Max daily drawdown: {max_dd:.2f}%")
        # Check if performance meets targets
        if (all(TARGET_WEEKLY_LOWER <= wp <= TARGET_WEEKLY_UPPER for wp in weekly_profits) and
            win_rate >= target_win_rate and
            max_dd <= MAX_DAILY_DRAWDOWN_PERCENT):
            print("Target performance achieved!")
            best_model = model
            break
        else:
            global RISK_FACTOR, WEEKLY_BONUS
            # UPDATED: More conservative risk updates to push toward a high win-rate strategy.
            RISK_FACTOR = min(RISK_FACTOR + 0.01, 0.15)
            WEEKLY_BONUS = min(WEEKLY_BONUS * 1.2, 1e6)
            print(f"Adaptive update: New RISK_FACTOR = {RISK_FACTOR:.2f}, New WEEKLY_BONUS = {WEEKLY_BONUS:.2f}")
        current_metric = avg_week_profit * win_rate
        if best_metrics is None or current_metric > best_metrics:
            best_metrics = current_metric
            best_iteration = iteration
            best_model = model
            print(f"New best model at iteration {iteration} with combined metric {best_metrics:.4f}")
        iteration += 1
        vec_train_env = DummyVecEnv([make_train_env])
        model.set_env(vec_train_env)
        time.sleep(1)
        if iteration >= 20:
            print("Reached maximum adaptive iterations.")
            break
    # Fine-tuning phase: further train best model with a lower learning rate.
    if best_model is not None:
        print("Starting fine-tuning phase on best model...")
        fine_tune_timesteps = 200_000
        model.learning_rate = 5e-5
        best_model.learn(total_timesteps=fine_tune_timesteps, reset_num_timesteps=False)
        print("Fine-tuning complete.")
    # Final testing on full test set
    test_env = GoldTradingEnv(df_test, initial_balance=INITIAL_BALANCE, verbose=True,
                              BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                              Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                              TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS,
                              xgb_model=xgb_model, render_mode='human')
    obs = test_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        total_reward += reward
        test_env.render(mode='human')
    print(f"\nFinal Test on Full Test Set:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final balance on test set: {test_env.current_balance:.2f}")
    print(f"Max daily drawdown on test set: {test_env.daily_drawdown_percent:.2f}%")
    win_rate_test = compute_win_rate(test_env.trade_log)
    if win_rate_test is not None:
        print(f"Win rate on test set: {win_rate_test*100:.2f}%")
    else:
        print("No trade closures to compute win rate.")
    trade_log_df = pd.DataFrame(test_env.trade_log)
    trade_log_df.to_csv("test_trade_log.csv", index=False)
    print("Saved full test trade log to test_trade_log.csv")
    final_backtest = backtest_multi_week(best_model, df_test, weeks=multi_week, initial_balance=INITIAL_BALANCE, xgb_model=xgb_model)
    if final_backtest is not None:
        print("Final multi-week backtest summary:", final_backtest)
    best_model.save("advanced_ppo_gold_trader")
    mt5.shutdown()
    sys.stdout.close()

if __name__ == "__main__":
    main()


    