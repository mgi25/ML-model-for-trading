import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import time

# Required libraries for RL and MT5 data
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
SYMBOL = "XAUUSDm"              # Symbol to trade
INITIAL_BALANCE = 100_000.0
MAX_DAILY_DRAWDOWN_PERCENT = 4.0
MAX_OVERALL_DRAWDOWN_PERCENT = 10.0
PARTIAL_CLOSE_RATIO = 0.5       # For partial trade closing
TRAIN_TIMEFRAMES_DAYS = 365     # 1 year of data

ITER_TIMESTEPS = 50_000         # Timesteps per training iteration
MAX_ITERATIONS = 20             # Maximum adaptive iterations

# Strategy parameters (inspired by your MT5 code)
BARS_N = 5                    # Lookback for breakout calculation
SH_INPUT = 0                  # Start Hour (0 means inactive)
EH_INPUT = 0                  # End Hour (0 means inactive)
TP_POINTS = 200               # Take profit points
SL_POINTS = 200               # Stop loss points
TSL_TRIGGER_POINTS = 15       # Points in profit before trailing stop activates
TSL_POINTS = 10               # Trailing stop points
_POINT = 0.01                 # Assumed point value

# RL risk parameters
RISK_FACTOR = 0.07            # Starting risk factor (fraction of balance risked)
REWARD_MULTIPLIER = 10.0      # Base per-step reward scaling
MAX_LOT_SIZE = 10.0           # Maximum allowed lot size

# Weekly reward shaping parameters (tuned for less dominating influence)
TARGET_WEEKLY_RETURN = 0.045  # Target weekly return (4.5%)
TARGET_WEEKLY_LOWER = 0.04    
TARGET_WEEKLY_UPPER = 0.05    
WEEKLY_COEFF = 200.0          # Lower continuous shaping coefficient
WEEKLY_BONUS = 2000.0         # Bonus if weekly return is within target
WEEKLY_PENALTY_MULTIPLIER = 30000.0  # Penalty multiplier if outside target

# Drawdown penalties
DAILY_DRAWDOWN_PENALTY = 1.0
OVERALL_DRAWDOWN_PENALTY = 2.0

# Trade closure reward adjustments
TRADE_BONUS = 30.0            # Bonus for profitable closure
TRADE_PENALTY = 10.0          # Penalty for losing trades

# Bonus for taking active (non-hold) actions
TRADE_FREQUENCY_BONUS = 1.0

# Direct profit reward weight (ties net profit percentage directly to reward)
PROFIT_REWARD_WEIGHT = 2.0

# ---------------- Technical Indicator Functions ----------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def dynamic_lot_size(current_balance, atr_value, risk_factor=RISK_FACTOR, factor=100):
    if atr_value < 1e-9:
        atr_value = 1.0
    risk_amount = risk_factor * current_balance
    lots = risk_amount / (atr_value * factor)
    return max(1.0, min(lots, MAX_LOT_SIZE))

# ---------------- MT5 CONNECTION & DATA COLLECTION ----------------
def initialize_mt5(account_id=None, password=None, server=None):
    if not mt5.initialize():
        print("MT5 Initialize() failed. Error code =", mt5.last_error())
        return False
    return True

def download_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, start_days=TRAIN_TIMEFRAMES_DAYS, num_bars=0):
    if num_bars <= 0:
        utc_now = datetime.now()
        utc_from = utc_now - timedelta(days=start_days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
    else:
        utc_now = datetime.now()
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
    # Add EMA50 and its slope as extra features
    df['ema50'] = compute_ema(df['close'], span=50)
    df['ema50_slope'] = df['ema50'].diff().fillna(0.0)
    df['ema50'] = df['ema50'].fillna(0.0)
    return df

# ---------------- CUSTOM GYM ENVIRONMENT ----------------
class GoldTradingEnv(gym.Env):
    """
    Custom Gym environment for RL-based gold trading on XAUUSDm.
    This environment integrates breakout scalping elements, trailing stops,
    extra technical features (EMA50 and its slope), and a reward function that includes:
      - Base incremental equity change
      - Trade closure bonus/penalty
      - Direct profit reward component
      - Continuous weekly shaping plus week-boundary bonus/penalty
      - A bonus for taking active (non-hold) actions.
    
    Observations (15 features):
      [open, high, low, close, volume, rsi, macd_hist, atr, position_size,
       current_balance, daily_drawdown%, overall_peak_balance, partial_close_flag,
       week_elapsed, weekly_return_so_far]
       
    Actions (discrete): 0=hold, 1=open long, 2=open short, 3=close partial, 4=close full.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=INITIAL_BALANCE, verbose=False,
                 BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                 Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                 TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS):
        super(GoldTradingEnv, self).__init__()
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.verbose = verbose

        # Store strategy parameters
        self.BarsN = BarsN
        self.SHInput = SHInput
        self.EHInput = EHInput
        self.Tppoints = Tppoints
        self.Slpoints = Slpoints
        self.TslTriggerPoints = TslTriggerPoints
        self.TslPoints = TslPoints

        # Compute technical indicators
        self.df['rsi'] = compute_rsi(self.df['close'], period=14).fillna(50.0)
        macd_line, signal_line, macd_hist = compute_macd(self.df['close'])
        self.df['macd_hist'] = macd_hist.fillna(0.0)
        self.df['atr'] = compute_atr(self.df, period=14).fillna(0.0)
        if 'ema50' not in self.df.columns:
            self.df['ema50'] = compute_ema(self.df['close'], span=50).fillna(0.0)
            self.df['ema50_slope'] = self.df['ema50'].diff().fillna(0.0)

        # Initialize trade state
        self.current_step = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.daily_peak_balance = initial_balance
        self.daily_drawdown_percent = 0.0
        self.overall_peak_balance = initial_balance
        self.partial_close_triggered = False
        self.trade_log = []
        self.prev_equity = self.equity
        self.stop_loss = None

        # Weekly tracking
        self.week_start_time = self.df.iloc[0]["time"]
        self.week_start_balance = self.current_balance

        # Observation space: 15 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
    
    def _compute_breakout_levels(self):
        idx = self.current_step
        start = max(0, idx - self.BarsN)
        end = min(len(self.df) - 1, idx + self.BarsN)
        local_high = self.df.loc[start:end, 'high'].max()
        local_low = self.df.loc[start:end, 'low'].min()
        return local_high, local_low

    def _get_extra_features(self):
        current_time = self.df.iloc[self.current_step]["time"]
        week_elapsed = (current_time - self.week_start_time).total_seconds() / (7 * 24 * 3600)
        week_elapsed = min(week_elapsed, 1.0)
        weekly_return_so_far = (self.current_balance / self.week_start_balance) - 1
        return week_elapsed, weekly_return_so_far

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        week_elapsed, weekly_return_so_far = self._get_extra_features()
        base_obs = np.array([
            row.open, row.high, row.low, row.close, row.volume, row.rsi,
            row.macd_hist, row.atr, self.position_size, self.current_balance,
            self.daily_drawdown_percent, self.overall_peak_balance, float(self.partial_close_triggered)
        ], dtype=np.float32)
        extra_obs = np.array([week_elapsed, weekly_return_so_far], dtype=np.float32)
        obs = np.concatenate((base_obs, extra_obs))
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        self.stop_loss = None
        self.week_start_time = self.df.iloc[0]["time"]
        self.week_start_balance = self.current_balance
        return self._get_observation()
    
    def _apply_trading_hours(self, current_time):
        if self.SHInput and current_time.hour < self.SHInput:
            return False
        if self.EHInput and self.EHInput != 0 and current_time.hour > self.EHInput:
            return False
        return True

    def _update_trailing_stop(self, current_price):
        if self.position_size == 0 or self.stop_loss is None:
            return
        if self.position_size > 0:
            profit_points = (current_price - self.entry_price) / _POINT
            if profit_points >= self.TslTriggerPoints:
                new_sl = current_price - self.TslPoints * _POINT
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
        elif self.position_size < 0:
            profit_points = (self.entry_price - current_price) / _POINT
            if profit_points >= self.TslTriggerPoints:
                new_sl = current_price + self.TslPoints * _POINT
                if new_sl < self.stop_loss:
                    self.stop_loss = new_sl

    def _check_stop_loss_hit(self, current_price):
        if self.position_size > 0 and self.stop_loss is not None:
            if current_price <= self.stop_loss:
                return True
        if self.position_size < 0 and self.stop_loss is not None:
            if current_price >= self.stop_loss:
                return True
        return False

    def _update_equity(self, price):
        if self.position_size == 0:
            self.equity = self.current_balance
        else:
            pip_value = self.position_size * (price - self.entry_price)
            self.equity = self.current_balance + pip_value
        if self.equity > self.daily_peak_balance:
            self.daily_peak_balance = self.equity
        if self.equity > self.overall_peak_balance:
            self.overall_peak_balance = self.equity
        self.daily_drawdown_percent = max((self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0, 0.0)
    
    def _open_position(self, lots, direction, price):
        self.position_size = lots if direction == "long" else -lots
        self.entry_price = price
        if direction == "long":
            self.stop_loss = price - self.Slpoints * _POINT
        else:
            self.stop_loss = price + self.Slpoints * _POINT

    def _close_partial(self, current_price):
        closed_size = self.position_size * PARTIAL_CLOSE_RATIO
        remaining_size = self.position_size - closed_size
        pip_value = (current_price - self.entry_price) if self.position_size >= 0 else (self.entry_price - current_price)
        realized_pnl = pip_value * closed_size
        self.current_balance += realized_pnl
        self.position_size = remaining_size if abs(remaining_size) >= 1e-6 else 0.0
        self.partial_close_triggered = True
        if self.position_size == 0:
            self.stop_loss = None
        bonus = self._trade_closure_bonus()
        return bonus

    def _close_full(self, current_price):
        if self.position_size == 0:
            return 0.0
        pip_value = (current_price - self.entry_price) if self.position_size >= 0 else (self.entry_price - current_price)
        realized_pnl = pip_value * abs(self.position_size)
        self.current_balance += realized_pnl
        self.position_size = 0.0
        self.entry_price = 0.0
        self.partial_close_triggered = False
        self.stop_loss = None
        bonus = self._trade_closure_bonus()
        return bonus

    def _trade_closure_bonus(self):
        trade_return = (self.current_balance - self.initial_balance) / self.initial_balance
        atr_value = self.df.iloc[self.current_step].atr
        close_price = self.df.iloc[self.current_step].close
        if atr_value < 1e-9 or close_price < 1e-9:
            return 0.0
        norm_atr = atr_value / close_price
        if trade_return > norm_atr:
            return TRADE_BONUS
        elif trade_return < -0.5 * norm_atr:
            return -TRADE_PENALTY
        return 0.0

    def _execute_action(self, action, current_price):
        current_time = self.df.iloc[self.current_step]["time"]
        if not self._apply_trading_hours(current_time):
            if self.position_size != 0:
                self._close_full(current_price)
            return
        self.closure_bonus = 0.0
        if action == 0:
            self.closure_bonus += TRADE_FREQUENCY_BONUS
        elif action == 1:
            if self.position_size == 0:
                local_high, _ = self._compute_breakout_levels()
                atr_value = self.df.iloc[self.current_step].atr
                lots = dynamic_lot_size(self.current_balance, atr_value)
                self._open_position(lots, "long", current_price)
        elif action == 2:
            if self.position_size == 0:
                _, local_low = self._compute_breakout_levels()
                atr_value = self.df.iloc[self.current_step].atr
                lots = dynamic_lot_size(self.current_balance, atr_value)
                self._open_position(lots, "short", current_price)
        elif action == 3:
            if self.position_size != 0 and not self.partial_close_triggered:
                self.closure_bonus = self._close_partial(current_price)
        elif action == 4:
            if self.position_size != 0:
                self.closure_bonus = self._close_full(current_price)

    def _compute_breakout_levels(self):
        idx = self.current_step
        start = max(0, idx - self.BarsN)
        end = min(len(self.df) - 1, idx + self.BarsN)
        local_high = self.df.loc[start:end, 'high'].max()
        local_low = self.df.loc[start:end, 'low'].min()
        return local_high, local_low

    def step(self, action):
        current_price = self.df.iloc[self.current_step].close
        old_equity = self.equity
        old_balance = self.current_balance
        old_position = self.position_size

        self._update_equity(current_price)
        self._execute_action(action, current_price)
        self._update_trailing_stop(current_price)
        if self._check_stop_loss_hit(current_price) and self.position_size != 0:
            self.closure_bonus = self._close_full(current_price)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if not done:
            next_price = self.df.iloc[self.current_step].close
            self._update_equity(next_price)
        
        reward = ((self.equity - old_equity) / self.initial_balance) * REWARD_MULTIPLIER
        reward += self.closure_bonus

        # Direct profit reward component
        profit_reward = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100.0 * PROFIT_REWARD_WEIGHT
        reward += profit_reward

        week_elapsed, weekly_return_so_far = self._get_extra_features()
        continuous_reward = -WEEKLY_COEFF * (weekly_return_so_far - TARGET_WEEKLY_RETURN)**2 * week_elapsed
        reward += continuous_reward

        current_time = self.df.iloc[self.current_step]["time"]
        if (current_time.isocalendar().year != self.week_start_time.isocalendar().year or
            current_time.isocalendar().week != self.week_start_time.isocalendar().week):
            if TARGET_WEEKLY_LOWER <= weekly_return_so_far <= TARGET_WEEKLY_UPPER:
                reward += WEEKLY_BONUS
            else:
                reward -= WEEKLY_PENALTY_MULTIPLIER * abs(weekly_return_so_far - TARGET_WEEKLY_RETURN)
            self.week_start_time = current_time
            self.week_start_balance = self.current_balance

        if self._calculate_daily_drawdown() > MAX_DAILY_DRAWDOWN_PERCENT:
            reward -= DAILY_DRAWDOWN_PENALTY
            done = True
        if self._calculate_overall_drawdown() > MAX_OVERALL_DRAWDOWN_PERCENT:
            reward -= OVERALL_DRAWDOWN_PENALTY
            done = True

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
        return self._get_observation(), reward, done, {}
    
    def _calculate_daily_drawdown(self):
        if self.daily_peak_balance == 0:
            return 0.0
        return max((self.daily_peak_balance - self.equity) / self.daily_peak_balance * 100.0, 0.0)
    
    def _calculate_overall_drawdown(self):
        if self.overall_peak_balance == 0:
            return 0.0
        return max((self.overall_peak_balance - self.equity) / self.overall_peak_balance * 100.0, 0.0)
    
    def render(self, mode='human'):
        pass

# ---------------- Utility: Compute Win Rate ----------------
def compute_win_rate(trade_log):
    closure_trades = [entry for entry in trade_log if entry.get("action") in [3, 4]]
    if not closure_trades:
        return None
    wins = sum(1 for entry in closure_trades if entry["new_balance"] > entry["old_balance"])
    return wins / len(closure_trades)

# ---------------- BACKTEST FUNCTION FOR LAST WEEK ----------------
def backtest_last_week(model, df_test, initial_balance=INITIAL_BALANCE):
    last_week = df_test['time'].apply(lambda t: t.isocalendar().week).max()
    df_last_week = df_test[df_test['time'].apply(lambda t: t.isocalendar().week) == last_week].copy()
    if df_last_week.empty:
        print("No data for the last week.")
        return None
    print(f"Backtesting on last week (ISO week {last_week})...")
    env_last_week = GoldTradingEnv(df_last_week, initial_balance=INITIAL_BALANCE, verbose=True,
                                   BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                                   Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                                   TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS)
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
    print(f"Win rate: {win_rate*100:.2f}%" if win_rate is not None else "No trade closures to compute win rate.")
    return {
        "final_balance": env_last_week.current_balance,
        "total_reward": total_reward,
        "max_daily_drawdown": env_last_week.daily_drawdown_percent,
        "win_rate": win_rate
    }

# ---------------- MAIN SCRIPT WITH ADAPTIVE TRAINING LOOP ----------------
def main():
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

    def make_train_env():
        return GoldTradingEnv(df_train, initial_balance=INITIAL_BALANCE, verbose=False,
                              BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                              Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                              TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS)
    vec_train_env = DummyVecEnv([make_train_env])

    policy_kwargs = dict(net_arch={"pi": [256,256,256], "vf": [256,256,256]})
    model = PPO(
        "MlpPolicy",
        vec_train_env,
        verbose=1,
        tensorboard_log="./ppo_xauusdm_tensorboard/",
        learning_rate=1e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device=device
    )

    target_weekly_profit = TARGET_WEEKLY_RETURN  # Target weekly profit (4.5%)
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
        print(f"Iteration {iteration} -- Weekly profit: {weekly_profit:.2%}")
        if weekly_profit >= target_weekly_profit:
            print("Target weekly profit reached. Stopping training loop.")
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
        print("Maximum adaptive iterations reached without meeting target weekly profit.")

    test_env = GoldTradingEnv(df_test, initial_balance=INITIAL_BALANCE, verbose=True,
                              BarsN=BARS_N, SHInput=SH_INPUT, EHInput=EH_INPUT,
                              Tppoints=TP_POINTS, Slpoints=SL_POINTS,
                              TslTriggerPoints=TSL_TRIGGER_POINTS, TslPoints=TSL_POINTS)
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
        print(f"Win rate on test set: {win_rate*100:.2f}%")
    else:
        print("No trade closures to compute win rate.")
    trade_log_df = pd.DataFrame(test_env.trade_log)
    trade_log_df.to_csv("test_trade_log.csv", index=False)
    print("Saved full test trade log to test_trade_log.csv")

    final_backtest = backtest_last_week(model, df_test, initial_balance=INITIAL_BALANCE)
    if final_backtest is not None:
        print("Final backtest (last week) summary:", final_backtest)

    model.save("ppo_gold_trader_xauusdm")
    mt5.shutdown()

if __name__ == "__main__":
    main()
