"""
================================================================================
Leverage CPPO - Constrained PPO (Lagrangian) for 30x Leveraged Scalping
================================================================================
기반: leverage_pro.py의 V2 환경 + Lagrangian Constrained PPO

핵심 개념: 보상(Reward)과 비용(Cost)을 분리
- Reward: Equity-Change + Hindsight (수익 극대화)
- Cost: 제약 위반 시 비용 발생 (리스크 관리)
- Lagrangian: effective_reward = reward - λ × cost
  → λ는 제약 위반에 따라 자동 조절

3가지 Cost 제약:
A. 드로다운 제약 (Drawdown Constraint)
   → 고점 대비 -10% 초과 시 cost 발생
   → SL_EXIT -73% 같은 대형 손실 원천 차단

B. 매매 빈도 제약 (Transaction Cost Constraint)
   → 손실 매매 시 cost 발생
   → 수수료 손실 매매를 물리적으로 억제

C. 사후 평가 연동 (Hindsight-based Cost)
   → 기회비용 상실 > 2% 시 cost 발생
   → "너무 일찍 파는 것"을 규칙 위반으로 간주

Lagrangian 작동 원리:
- λ = 0에서 시작 (일반 PPO와 동일)
- 제약 위반 시 λ 증가 → cost 페널티 강화
- 제약 준수 시 λ 감소 → 수익 극대화에 집중
- 자동 균형: 리스크 관리 ↔ 수익 극대화

데이터/환경: leverage_pro.py와 동일 (Multi-Symbol × Multi-Timeframe)
================================================================================
"""

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch as th
import platform
import time
import os

# ==========================================
# 1. 설정
# ==========================================
TRAIN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"
]
TEST_SYMBOL = "ADAUSDT"

MODEL_TYPE = "LSTM"

RUN_ID = "CPPO_Stage1"
INITIALIZE_FROM = None

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SAVE_DIR = os.path.join(RUNS_DIR, RUN_ID)
CHECKPOINT_FREQ = 100000

# 사후 평가 (V2 기반)
USE_HINDSIGHT = True
HINDSIGHT_STEPS = 20
HINDSIGHT_SCALE = 0.5       # 0.3→0.5: "더 버텨서 큰 수익"이 유리하도록 강화

# ==========================================
# CPPO 제약 설정
# ==========================================
# A. 드로다운 제약: 고점 대비 N% 초과 시 cost 발생
DRAWDOWN_COST_THRESHOLD = 0.10    # 10% 드로다운 초과 시
DRAWDOWN_COST_SCALE = 5.0         # cost 증폭 계수

# B. 매매 빈도 제약: 손실 매매 시 cost 발생
MIN_PROFIT_PCT = 0.0              # 0% 미만 = 손실 매매

# C. 사후 평가 연동: 기회비용 > N% 시 cost 발생
HINDSIGHT_COST_THRESHOLD = 0.02   # 2% 초과 기회비용

# Lagrangian 승수 설정
COST_LIMIT = 0.20                 # 에피소드 평균 cost 상한 (0.15→0.20: 추세 추종 여유)
LAMBDA_LR = 0.05                  # λ 학습률
LAMBDA_MAX = 10.0                 # λ 최대값
LAMBDA_UPDATE_FREQ = 20           # λ 업데이트 주기 (에피소드)


# ==========================================
# 2. 데이터 수집 (leverage_pro.py와 동일)
# ==========================================
def fetch_raw_ohlcv(exchange, symbol, timeframe, total_limit):
    all_ohlcv = []
    last_ts = None
    while len(all_ohlcv) < total_limit:
        limit = min(1000, total_limit - len(all_ohlcv))
        params = {'endTime': last_ts - 1} if last_ts else {}
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        if not ohlcv:
            break
        all_ohlcv = ohlcv + all_ohlcv
        last_ts = ohlcv[0][0]
    return pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


def compute_indicators(df, prefix=''):
    p = prefix
    df[f'{p}returns'] = df['close'].pct_change()
    ma7 = df['close'].rolling(7).mean()
    ma25 = df['close'].rolling(25).mean()
    df[f'{p}volatility'] = df[f'{p}returns'].rolling(20).std()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df[f'{p}rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df[f'{p}bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()

    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    df[f'{p}ma7_ratio'] = (df['close'] - ma7) / (ma7 + 1e-9)
    df[f'{p}ma25_ratio'] = (df['close'] - ma25) / (ma25 + 1e-9)
    df[f'{p}macd_ratio'] = macd / (df['close'] + 1e-9)
    df[f'{p}macd_signal_ratio'] = macd_signal / (df['close'] + 1e-9)
    df[f'{p}atr_ratio'] = atr / (df['close'] + 1e-9)

    feature_cols = [
        f'{p}returns', f'{p}ma7_ratio', f'{p}ma25_ratio', f'{p}volatility',
        f'{p}rsi', f'{p}bb_pct', f'{p}macd_ratio', f'{p}macd_signal_ratio', f'{p}atr_ratio'
    ]
    return df, feature_cols


def fetch_symbol_data(exchange, symbol, total_limit=10000):
    print(f"   📡 {symbol}", end=" ")
    df_5m = fetch_raw_ohlcv(exchange, symbol, "5m", total_limit)
    df_5m, feat_5m = compute_indicators(df_5m, prefix='')

    h1_limit = total_limit // 12 + 200
    df_1h = fetch_raw_ohlcv(exchange, symbol, "1h", h1_limit)
    df_1h, feat_1h = compute_indicators(df_1h, prefix='h1_')

    df_1h_merge = df_1h[['timestamp'] + feat_1h].copy()
    df_1h_merge['ts_1h_close'] = df_1h_merge['timestamp'] + 3600000
    df_1h_merge = df_1h_merge.drop(columns=['timestamp'])

    df_merged = pd.merge_asof(
        df_5m.sort_values('timestamp'),
        df_1h_merge.sort_values('ts_1h_close'),
        left_on='timestamp', right_on='ts_1h_close',
        direction='backward'
    )

    all_feat_cols = feat_5m + feat_1h
    nan_mask = df_merged[all_feat_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    df_clean = df_merged[~nan_mask].reset_index(drop=True)

    price_array = df_clean[['close']].values
    tech_array = df_clean[all_feat_cols].values
    print(f"→ 5m:{len(df_5m)} + 1h:{len(df_1h)} → {len(df_clean)}개 (NaN제거:{nan_count}) [피처:{tech_array.shape[1]}]")
    return price_array, tech_array, all_feat_cols


# ==========================================
# 3. Equity-Change Reward (leverage_pro.py와 동일)
# ==========================================
class EquityChangeRewardCalculator:
    """
    reward = delta(equity) / initial_equity + 추가 신호 + Sharpe 보너스

    Sharpe Ratio 강화:
    - 매 스텝 equity_change를 기록하여 rolling Sharpe Ratio 계산
    - 에피소드 종료 시 Sharpe 기반 보너스/페널티 부여
    - 변동성 대비 수익이 높을수록 보상 증가
    - "큰 수익 + 큰 변동" < "적당한 수익 + 낮은 변동"
    """

    def __init__(self, leverage=30, sharpe_window=100, sharpe_scale=0.2):
        self.leverage = leverage
        self.initial_equity = 100.0
        self.prev_equity = 100.0
        self.sharpe_window = sharpe_window   # rolling window 크기
        self.sharpe_scale = sharpe_scale     # Sharpe 보너스 가중치
        self.equity_changes = []             # 스텝별 equity 변화율 기록

    def reset(self, initial_equity):
        self.initial_equity = initial_equity
        self.prev_equity = initial_equity
        self.equity_changes = []

    def _calculate_sharpe(self):
        """최근 N스텝의 Rolling Sharpe Ratio 계산"""
        if len(self.equity_changes) < 20:
            return 0.0
        recent = self.equity_changes[-self.sharpe_window:]
        mean_ret = np.mean(recent)
        std_ret = np.std(recent)
        if std_ret < 1e-9:
            return 0.0
        return mean_ret / std_ret

    def calculate_reward(self, curr_equity, is_liquidated=False,
                         is_sl_exit=False, has_position=True,
                         is_episode_end=False, survived=True):
        equity_change = (curr_equity - self.prev_equity) / self.initial_equity
        reward = equity_change

        # Sharpe용 수익률 기록
        self.equity_changes.append(equity_change)

        # 매 스텝 Sharpe 미세 조정: 변동성이 클 때 약한 페널티
        if len(self.equity_changes) >= 50:
            recent_std = np.std(self.equity_changes[-50:])
            # 변동성이 높으면 약한 음수 신호 (과도한 리스크 억제)
            reward -= recent_std * 0.01

        if is_liquidated:
            reward -= 0.5
        if is_sl_exit:
            reward -= 0.15
        if not has_position:
            reward += 0.0002

        # 에피소드 종료: ROI 보너스 + Sharpe 보너스
        if is_episode_end and survived:
            roi = (curr_equity - self.initial_equity) / self.initial_equity
            reward += np.clip(roi, -0.3, 0.3)

            # Sharpe Ratio 보너스: 안정적 수익일수록 추가 보상
            sharpe = self._calculate_sharpe()
            sharpe_bonus = np.clip(sharpe, -1.0, 1.0) * self.sharpe_scale
            reward += sharpe_bonus

        self.prev_equity = curr_equity
        return reward


# ==========================================
# 4. 기본 환경 V1 (leverage_pro.py와 동일)
# ==========================================
class LeverageProEnv(gym.Env):
    """30x Isolated Margin Scalping Environment (V1 base)"""

    def __init__(self, datasets, stats,
                 leverage=35, initial_capital=100.0, entry_size=5.0):
        super().__init__()
        self.datasets = datasets
        self.stats = stats
        self.leverage = leverage
        self.initial_capital = initial_capital
        self.entry_size = entry_size
        self.taker_fee = 0.00055
        self.liquidation_pct = 1.0 / leverage

        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.03
        self.trailing_activation = 0.02
        self.trailing_callback = 0.005
        self.trade_cooldown = 5

        self.reward_calc = EquityChangeRewardCalculator(leverage=leverage)
        self.price_array, self.tech_array = self.datasets[0]

        self.state_dim = self.tech_array.shape[1] + 4
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.max_step = len(self.price_array) - 2
        self.reset()

    def _normalize_tech(self, tech):
        normalized = (tech - self.stats['tech_mean']) / self.stats['tech_std']
        return np.clip(normalized, -3, 3)

    def _get_obs(self):
        tech = self._normalize_tech(self.tech_array[self.time])
        position_side = np.sign(self.position) if self.position != 0 else 0
        position_pnl = self.unrealized_pnl / self.entry_size if self.position != 0 else 0
        holding_time = (self.time - self.entry_time) / 100 if self.position != 0 else 0
        cash_ratio = self.cash / self.initial_capital
        state = np.concatenate([tech, [position_side, position_pnl, holding_time, cash_ratio]])
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        idx = np.random.randint(len(self.datasets))
        self.price_array, self.tech_array = self.datasets[idx]
        self.max_step = len(self.price_array) - 2

        self.time = 0
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        self.unrealized_pnl = 0
        self.total_asset = self.cash

        self.best_price_pct = 0
        self.trailing_active = False
        self.trades = []
        self.prev_action = 0
        self.last_trade_time = -self.trade_cooldown
        self.reward_calc.reset(self.initial_capital)
        return self._get_obs(), {}

    def _close_position(self, reason="EXIT"):
        curr_price = self.price_array[self.time][0]
        fee = abs(self.position) * self.taker_fee
        realized_pnl = self.unrealized_pnl - fee
        self.cash += self.entry_size + realized_pnl
        pnl_pct = (realized_pnl / self.entry_size) * 100
        duration = self.time - self.entry_time

        self.trades.append({
            'step': self.time, 'type': reason, 'price': curr_price,
            'pnl': realized_pnl, 'pnl_pct': pnl_pct, 'duration': duration
        })
        self.position = 0
        self.entry_price = 0
        self.trailing_active = False
        self.best_price_pct = 0
        return pnl_pct, duration

    def _open_position(self, action):
        curr_price = self.price_array[self.time][0]
        pos_value = self.entry_size * self.leverage
        fee = pos_value * self.taker_fee
        self.cash -= (self.entry_size + fee)
        self.position = pos_value if action > 0 else -pos_value
        self.entry_price = curr_price
        self.entry_time = self.time
        self.trades.append({
            'step': self.time, 'type': 'LONG' if action > 0 else 'SHORT',
            'price': curr_price
        })

    def step(self, actions):
        self.time += 1
        curr_price = self.price_array[self.time][0]
        action = float(actions[0])
        pnl_pct = 0
        is_liquidated = False
        is_sl_exit = False

        if self.position != 0:
            price_change = (curr_price - self.entry_price) / self.entry_price
            if self.position < 0:
                price_change = -price_change
            self.unrealized_pnl = abs(self.position) * price_change

            if price_change <= -self.liquidation_pct:
                pnl_pct, _ = self._close_position("LIQUIDATION")
                is_liquidated = True
            elif price_change <= -self.stop_loss_pct:
                pnl_pct, _ = self._close_position("SL_EXIT")
                is_sl_exit = True
            elif price_change >= self.take_profit_pct:
                pnl_pct, _ = self._close_position("TP_EXIT")
            elif self.trailing_active:
                if price_change > self.best_price_pct:
                    self.best_price_pct = price_change
                elif self.best_price_pct - price_change >= self.trailing_callback:
                    pnl_pct, _ = self._close_position("TS_EXIT")
            else:
                if price_change >= self.trailing_activation:
                    self.trailing_active = True
                    self.best_price_pct = price_change
                if abs(action) < 0.3 or (action * self.position < 0):
                    pnl_pct, _ = self._close_position("AI_EXIT")

        line = 0.8
        cooldown_ok = (self.time - self.last_trade_time) >= self.trade_cooldown
        if self.position == 0 and abs(action) > line and self.cash >= self.entry_size and cooldown_ok:
            self._open_position(action)
            self.last_trade_time = self.time

        self.total_asset = self.cash + (self.entry_size if self.position != 0 else 0) + self.unrealized_pnl
        is_alive = self.total_asset >= self.initial_capital * 0.40
        is_episode_end = self.time >= self.max_step
        done = is_episode_end or not is_alive

        reward = self.reward_calc.calculate_reward(
            curr_equity=self.total_asset, is_liquidated=is_liquidated,
            is_sl_exit=is_sl_exit, has_position=(self.position != 0),
            is_episode_end=is_episode_end, survived=(is_alive and is_episode_end)
        )
        self.prev_action = action
        return self._get_obs(), reward, done, False, {
            'total_asset': self.total_asset, 'pnl_pct': pnl_pct
        }


# ==========================================
# 4-1. V2: 사후 평가 환경 (leverage_pro.py와 동일)
# ==========================================
class LeverageProEnvV2(LeverageProEnv):
    """V2: 포지션 종료 후 향후 N스텝 가격 참조하여 사후 평가 보상 추가"""

    def __init__(self, datasets, stats, leverage=35, initial_capital=100.0,
                 entry_size=5.0, hindsight_steps=20, hindsight_scale=0.3):
        self.hindsight_steps = hindsight_steps
        self.hindsight_scale = hindsight_scale
        self._last_exit_happened = False
        self._last_exit_price = 0.0
        self._last_exit_side = 0
        super().__init__(datasets, stats, leverage, initial_capital, entry_size)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._last_exit_happened = False
        self._last_exit_price = 0.0
        self._last_exit_side = 0
        return obs, info

    def _close_position(self, reason="EXIT"):
        self._last_exit_price = self.price_array[self.time][0]
        self._last_exit_side = 1 if self.position > 0 else -1
        self._last_exit_happened = True
        return super()._close_position(reason)

    def _calculate_hindsight_reward(self, exit_price, position_side):
        future_start = self.time + 1
        future_end = min(self.time + self.hindsight_steps + 1, len(self.price_array))
        if future_end <= future_start:
            return 0.0

        future_prices = self.price_array[future_start:future_end, 0]

        if position_side > 0:
            best_favorable = np.max(future_prices)
            worst_adverse = np.min(future_prices)
        else:
            best_favorable = np.min(future_prices)
            worst_adverse = np.max(future_prices)

        if position_side > 0:
            missed_gain_pct = (best_favorable - exit_price) / exit_price
        else:
            missed_gain_pct = (exit_price - best_favorable) / exit_price

        opportunity_penalty = -max(missed_gain_pct, 0)

        if position_side > 0:
            avoided_loss_pct = (exit_price - worst_adverse) / exit_price
        else:
            avoided_loss_pct = (worst_adverse - exit_price) / exit_price

        risk_bonus = max(avoided_loss_pct, 0)
        scale = self.leverage * (self.entry_size / self.initial_capital)
        return (opportunity_penalty + risk_bonus) * scale * self.hindsight_scale

    def step(self, actions):
        self._last_exit_happened = False
        obs, reward, done, truncated, info = super().step(actions)
        if self._last_exit_happened:
            hindsight = self._calculate_hindsight_reward(
                self._last_exit_price, self._last_exit_side
            )
            reward += hindsight
            info['hindsight_reward'] = hindsight
        return obs, reward, done, truncated, info


# ==========================================
# 4-2. V3: CPPO 환경 (Constrained PPO 전용)
# ==========================================
class LeverageProEnvV3(LeverageProEnvV2):
    """
    V3: Constrained PPO 환경 - Lagrangian Cost 제약

    V2의 Reward + Hindsight를 그대로 유지하면서,
    3가지 Cost 신호를 추가로 계산합니다.

    Cost는 reward와 별도로 계산되며, Lagrangian 승수(λ)를 통해
    effective_reward = reward - λ × cost 로 변환됩니다.

    A. 드로다운 제약 (Drawdown Constraint)
       고점 대비 drawdown이 threshold(10%)를 넘으면 cost 발생
       → 큰 손실이 누적되기 전에 포지션 정리를 유도

    B. 매매 빈도 제약 (Transaction Cost Constraint)
       매매 수익이 수수료를 충당하지 못하면 cost 발생
       → 잦은 손실 매매를 물리적으로 억제

    C. 사후 평가 연동 (Hindsight-based Cost)
       기회비용 상실이 threshold(2%)를 넘으면 cost 발생
       → "너무 일찍 파는 것"을 규칙 위반으로 간주
    """

    def __init__(self, datasets, stats, leverage=35, initial_capital=100.0,
                 entry_size=5.0, hindsight_steps=20, hindsight_scale=0.3,
                 drawdown_threshold=0.10, drawdown_cost_scale=5.0,
                 min_profit_pct=0.0, hindsight_cost_threshold=0.02):
        # Cost 설정
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_cost_scale = drawdown_cost_scale
        self.min_profit_pct = min_profit_pct
        self.hindsight_cost_threshold = hindsight_cost_threshold

        # Lagrangian 승수 (LagrangianCallback이 동적으로 조절)
        self.cost_lambda = 0.0

        # Cost 추적
        self._peak_equity = initial_capital
        self._episode_cost = 0.0
        self._last_missed_gain_pct = 0.0

        super().__init__(datasets, stats, leverage, initial_capital,
                         entry_size, hindsight_steps, hindsight_scale)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._peak_equity = self.initial_capital
        self._episode_cost = 0.0
        self._last_missed_gain_pct = 0.0
        return obs, info

    def _calculate_hindsight_reward(self, exit_price, position_side):
        """V2 사후 평가 + Cost C용 기회비용(missed_gain_pct) 기록"""
        future_start = self.time + 1
        future_end = min(self.time + self.hindsight_steps + 1, len(self.price_array))
        if future_end <= future_start:
            self._last_missed_gain_pct = 0.0
            return 0.0

        future_prices = self.price_array[future_start:future_end, 0]

        if position_side > 0:
            best_favorable = np.max(future_prices)
            worst_adverse = np.min(future_prices)
        else:
            best_favorable = np.min(future_prices)
            worst_adverse = np.max(future_prices)

        if position_side > 0:
            missed_gain_pct = (best_favorable - exit_price) / exit_price
        else:
            missed_gain_pct = (exit_price - best_favorable) / exit_price

        # Cost C에서 사용할 기회비용 저장
        self._last_missed_gain_pct = max(missed_gain_pct, 0)

        opportunity_penalty = -max(missed_gain_pct, 0)

        if position_side > 0:
            avoided_loss_pct = (exit_price - worst_adverse) / exit_price
        else:
            avoided_loss_pct = (worst_adverse - exit_price) / exit_price

        risk_bonus = max(avoided_loss_pct, 0)
        scale = self.leverage * (self.entry_size / self.initial_capital)
        return (opportunity_penalty + risk_bonus) * scale * self.hindsight_scale

    def step(self, actions):
        # V2.step() 호출: base reward + hindsight
        # (V2.step 내부에서 _last_exit_happened 리셋 후 V1.step 실행)
        obs, reward, done, truncated, info = super().step(actions)

        # === Cost 계산 ===
        cost_a = 0.0  # 드로다운
        cost_b = 0.0  # 매매 손실
        cost_c = 0.0  # 기회비용

        # A. 드로다운 제약: 고점 대비 초과 하락분에 cost 부과
        self._peak_equity = max(self._peak_equity, self.total_asset)
        drawdown = (self._peak_equity - self.total_asset) / self._peak_equity
        if drawdown > self.drawdown_threshold:
            cost_a = (drawdown - self.drawdown_threshold) * self.drawdown_cost_scale

        # B. 매매 빈도 제약: 손실 매매 시 cost 부과
        if self._last_exit_happened:
            pnl_pct = info.get('pnl_pct', 0)
            if pnl_pct < self.min_profit_pct:
                cost_b = abs(self.min_profit_pct - pnl_pct) / 100

        # C. 사후 평가 연동: 기회비용 > 임계값 시 cost 부과
        if self._last_exit_happened and self._last_missed_gain_pct > self.hindsight_cost_threshold:
            cost_c = (self._last_missed_gain_pct - self.hindsight_cost_threshold)

        step_cost = cost_a + cost_b + cost_c
        self._episode_cost += step_cost

        # Lagrangian: effective_reward = reward - λ × cost
        reward -= self.cost_lambda * step_cost

        # info에 cost 정보 기록 (콜백에서 모니터링용)
        info['cost'] = step_cost
        info['cost_a'] = cost_a
        info['cost_b'] = cost_b
        info['cost_c'] = cost_c
        info['episode_cost'] = self._episode_cost
        info['cost_lambda'] = self.cost_lambda

        return obs, reward, done, truncated, info


# ==========================================
# 5. Lagrangian 콜백 (CPPO 핵심)
# ==========================================
class LagrangianCallback(BaseCallback):
    """
    Lagrangian 승수(λ) 동적 조절 콜백

    작동 원리:
    1. 매 에피소드 종료 시 episode_cost 기록
    2. 최근 N개 에피소드의 평균 cost 계산
    3. avg_cost > cost_limit → λ 증가 (제약 강화)
       avg_cost < cost_limit → λ 감소 (제약 완화)
    4. λ ∈ [0, lambda_max] 범위로 클리핑

    제약이 만족되면 λ → 0으로 수렴 (일반 PPO와 동일 동작)
    제약이 위반되면 λ 증가로 cost 페널티 강화
    """

    def __init__(self, cost_limit=0.15, lambda_lr=0.05,
                 lambda_update_freq=20, lambda_max=10.0, verbose=1):
        super().__init__(verbose)
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.lambda_update_freq = lambda_update_freq
        self.lambda_max = lambda_max

        # 에피소드별 추적
        self.episode_costs = []
        self.episode_rewards = []
        self.episode_costs_a = []
        self.episode_costs_b = []
        self.episode_costs_c = []
        self.lambda_history = []

        # 현재 에피소드 누적
        self._current_cost = 0.0
        self._current_reward = 0.0
        self._current_cost_a = 0.0
        self._current_cost_b = 0.0
        self._current_cost_c = 0.0

    def _get_inner_env(self):
        """VecNormalize → DummyVecEnv → LeverageProEnvV3"""
        return self.training_env.venv.envs[0]

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]

        self._current_cost += info.get('cost', 0.0)
        self._current_reward += reward
        self._current_cost_a += info.get('cost_a', 0.0)
        self._current_cost_b += info.get('cost_b', 0.0)
        self._current_cost_c += info.get('cost_c', 0.0)

        if self.locals['dones'][0]:
            self.episode_costs.append(self._current_cost)
            self.episode_rewards.append(self._current_reward)
            self.episode_costs_a.append(self._current_cost_a)
            self.episode_costs_b.append(self._current_cost_b)
            self.episode_costs_c.append(self._current_cost_c)

            self._current_cost = 0.0
            self._current_reward = 0.0
            self._current_cost_a = 0.0
            self._current_cost_b = 0.0
            self._current_cost_c = 0.0

            # λ 업데이트 (N 에피소드마다)
            if len(self.episode_costs) % self.lambda_update_freq == 0:
                self._update_lambda()

        return True

    def _update_lambda(self):
        if len(self.episode_costs) < self.lambda_update_freq:
            return

        recent_costs = self.episode_costs[-self.lambda_update_freq:]
        avg_cost = np.mean(recent_costs)

        # 제약 위반량: 양수면 위반, 음수면 준수
        constraint_violation = avg_cost - self.cost_limit

        inner_env = self._get_inner_env()
        old_lambda = inner_env.cost_lambda

        # Gradient ascent on λ (dual variable)
        new_lambda = old_lambda + self.lambda_lr * constraint_violation
        new_lambda = float(np.clip(new_lambda, 0.0, self.lambda_max))
        inner_env.cost_lambda = new_lambda

        self.lambda_history.append(new_lambda)

        if self.verbose > 0:
            avg_reward = np.mean(self.episode_rewards[-self.lambda_update_freq:])
            avg_a = np.mean(self.episode_costs_a[-self.lambda_update_freq:])
            avg_b = np.mean(self.episode_costs_b[-self.lambda_update_freq:])
            avg_c = np.mean(self.episode_costs_c[-self.lambda_update_freq:])
            ep = len(self.episode_costs)
            print(f"\n   [CPPO λ Update] Episode {ep}")
            print(f"   λ: {old_lambda:.4f} → {new_lambda:.4f}")
            print(f"   avg_cost: {avg_cost:.4f} (limit: {self.cost_limit:.4f})")
            print(f"   cost_A(drawdown): {avg_a:.4f} | "
                  f"cost_B(transaction): {avg_b:.4f} | "
                  f"cost_C(hindsight): {avg_c:.4f}")
            print(f"   avg_reward: {avg_reward:.4f}")


# ==========================================
# 6. 모델 아키텍처 (leverage_pro.py와 동일)
# ==========================================
MLP_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    activation_fn=th.nn.LeakyReLU,
    log_std_init=-0.55,
)

LSTM_POLICY_KWARGS = dict(
    lstm_hidden_size=128,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
    net_arch=dict(pi=[128], vf=[128]),
    activation_fn=th.nn.LeakyReLU,
    log_std_init=-0.55,
)

PPO_HYPERPARAMS = dict(
    n_steps=2048,
    batch_size=128,
    n_epochs=15,
    gamma=0.98,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)


def create_model(env, model_type="LSTM", initialize_from=None):
    """모델 생성 또는 이전 학습에서 로드"""
    if initialize_from is not None:
        prev_dir = os.path.join(RUNS_DIR, initialize_from)
        prev_model_path = os.path.join(prev_dir, "best_model.zip")

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"초기화 모델을 찾을 수 없습니다: {prev_model_path}\n"
                f"'{initialize_from}' 학습이 완료되었는지 확인하세요."
            )

        if model_type == "LSTM":
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(prev_model_path, env=env)
            print(f"\n🧠 모델: RecurrentPPO (LSTM)")
        else:
            model = PPO.load(prev_model_path, env=env)
            print(f"\n🧠 모델: PPO (MLP)")

        print(f"   📂 initialize-from: {initialize_from}")
        print(f"   📂 로드 경로: {prev_model_path}")

        prev_vecnorm_path = os.path.join(prev_dir, "vecnormalize.pkl")
        if os.path.exists(prev_vecnorm_path):
            env = VecNormalize.load(prev_vecnorm_path, env.venv)
            model.set_env(env)
            print(f"   📂 VecNormalize 통계 로드 완료")

        return model

    if model_type == "LSTM":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            policy_kwargs=LSTM_POLICY_KWARGS,
            learning_rate=1e-4,
            **PPO_HYPERPARAMS
        )
        print(f"\n🧠 모델: RecurrentPPO (LSTM) — CPPO")
        print(f"   Input(22) → LSTM(128) → Dense(128) → LeakyReLU → Action(1)")
    else:
        model = PPO(
            "MlpPolicy", env,
            policy_kwargs=MLP_POLICY_KWARGS,
            learning_rate=2e-4,
            **PPO_HYPERPARAMS
        )
        print(f"\n🧠 모델: PPO (MLP) — CPPO")
        print(f"   Input(22) → Dense(256) → LeakyReLU → Dense(128) → LeakyReLU → Action(1)")

    return model


# ==========================================
# 7. 데이터 로드
# ==========================================
print("=" * 60)
print("📊 Multi-Symbol, Multi-Timeframe 데이터 수집")
print(f"   학습 종목: {', '.join(TRAIN_SYMBOLS)}")
print(f"   테스트 종목: {TEST_SYMBOL}")
print("=" * 60)

exchange = ccxt.bybit()
all_symbol_data = {}

for sym in sorted(set(TRAIN_SYMBOLS + [TEST_SYMBOL])):
    all_symbol_data[sym] = fetch_symbol_data(exchange, sym, total_limit=60000)
    time.sleep(1)

train_datasets = []
all_train_techs = []

for sym in TRAIN_SYMBOLS:
    prices, techs, feat_cols = all_symbol_data[sym]
    n = int(len(prices) * 0.8)
    train_datasets.append((prices[:n], techs[:n]))
    all_train_techs.append(techs[:n])

test_prices_full, test_techs_full, _ = all_symbol_data[TEST_SYMBOL]
test_split = int(len(test_prices_full) * 0.8)
test_prices = test_prices_full[test_split:]
test_techs = test_techs_full[test_split:]

all_train_combined = np.concatenate(all_train_techs, axis=0)
data_stats = {
    'tech_mean': all_train_combined.mean(axis=0),
    'tech_std': all_train_combined.std(axis=0) + 1e-9
}

total_train_samples = sum(len(t) for _, t in train_datasets)
print(f"\n   ✅ 학습: {len(TRAIN_SYMBOLS)}종목 × 80% = {total_train_samples:,}개")
print(f"   ✅ 테스트: {TEST_SYMBOL} 20% = {len(test_prices):,}개")

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 8. CPPO 학습
# ==========================================
print("\n" + "=" * 60)
print("🛡️ CPPO - Constrained PPO (Lagrangian Relaxation)")
print("=" * 60)
print(f"📊 학습: {len(TRAIN_SYMBOLS)}종목 ({', '.join(TRAIN_SYMBOLS)})")
print(f"📊 테스트: {TEST_SYMBOL}")
print(f"📈 학습 데이터: {total_train_samples:,}개 / 테스트: {len(test_prices):,}개")
print(f"📐 피처: 5m(9) + 1h(9) = 18 + 포지션(4) = 22차원 상태")
print(f"🧠 모델 아키텍처: {MODEL_TYPE}")
print(f"⚡ 레버리지: 30x (Isolated)")
print(f"💾 RUN_ID: {RUN_ID}" + (f" (initialize-from: {INITIALIZE_FROM})" if INITIALIZE_FROM else ""))

print(f"\n🛡️ CPPO 제약 설정:")
print(f"   A. 드로다운 제약: >{DRAWDOWN_COST_THRESHOLD*100:.0f}% 초과 시 cost "
      f"(scale={DRAWDOWN_COST_SCALE})")
print(f"   B. 매매 손실 제약: <{MIN_PROFIT_PCT:.1f}% 수익 시 cost")
print(f"   C. 기회비용 제약: >{HINDSIGHT_COST_THRESHOLD*100:.0f}% 상실 시 cost")
print(f"   λ 설정: limit={COST_LIMIT}, lr={LAMBDA_LR}, "
      f"max={LAMBDA_MAX}, update_freq={LAMBDA_UPDATE_FREQ}")

print(f"\n💡 보상 함수 구성 (Equity-Change + Sharpe + Hindsight + CPPO):")
print("   - 매 스텝: delta(equity) / initial_equity — 실제 자산 변화 직결")
print("   - 매 스텝: 변동성 페널티 (-std × 0.01) — 과도한 리스크 억제")
print(f"   - 사후 평가(Hindsight): 기회비용/리스크회피 보상 (scale={HINDSIGHT_SCALE})")
print("   - 에피소드 종료: Sharpe Ratio 보너스 (안정적 수익 유도)")
print("   - CPPO: reward -= λ × cost (λ 자동 조절)")
print("   - VecNormalize: 자동 보상 스케일링")
print("=" * 60)

# V3 CPPO 환경 생성
def make_train_env():
    return LeverageProEnvV3(
        train_datasets, data_stats, leverage=30,
        hindsight_steps=HINDSIGHT_STEPS, hindsight_scale=HINDSIGHT_SCALE,
        drawdown_threshold=DRAWDOWN_COST_THRESHOLD,
        drawdown_cost_scale=DRAWDOWN_COST_SCALE,
        min_profit_pct=MIN_PROFIT_PCT,
        hindsight_cost_threshold=HINDSIGHT_COST_THRESHOLD
    )

print(f"\n🔬 환경: LeverageProEnvV3 (CPPO + Hindsight)")

train_env = DummyVecEnv([make_train_env])
train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                         clip_reward=10.0, gamma=0.98)

model = create_model(train_env, MODEL_TYPE, initialize_from=INITIALIZE_FROM)
total_timesteps_num = 1000000

os.makedirs(SAVE_DIR, exist_ok=True)

# 콜백: Lagrangian + 체크포인트
lagrangian_cb = LagrangianCallback(
    cost_limit=COST_LIMIT,
    lambda_lr=LAMBDA_LR,
    lambda_update_freq=LAMBDA_UPDATE_FREQ,
    lambda_max=LAMBDA_MAX,
    verbose=1,
)
checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=os.path.join(SAVE_DIR, "checkpoints"),
    name_prefix=RUN_ID,
    save_vecnormalize=True,
)

init_msg = f" (initialize-from: {INITIALIZE_FROM})" if INITIALIZE_FROM else " (새로 시작)"
print(f"\n🏋️ CPPO 학습 시작...{init_msg}")
print(f"   💾 저장 경로: {SAVE_DIR}")
print(f"   💾 체크포인트: {CHECKPOINT_FREQ:,} 스텝마다 자동 저장")
print(f"   🛡️ λ = 0.0 에서 시작 (제약 위반 시 자동 증가)")

model.learn(total_timesteps=total_timesteps_num,
            callback=[lagrangian_cb, checkpoint_cb])

# 최종 모델 저장
model.save(os.path.join(SAVE_DIR, "best_model"))
train_env.save(os.path.join(SAVE_DIR, "vecnormalize.pkl"))
print(f"\n💾 최종 모델 저장 완료:")
print(f"   📂 {os.path.join(SAVE_DIR, 'best_model.zip')}")
print(f"   📂 {os.path.join(SAVE_DIR, 'vecnormalize.pkl')}")


# ==========================================
# 9. 학습 곡선 (Reward + Cost + Lambda)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. 에피소드 보상
ax1 = axes[0, 0]
ax1.plot(lagrangian_cb.episode_rewards, alpha=0.5, color='blue', lw=0.5)
if len(lagrangian_cb.episode_rewards) > 50:
    window = min(50, len(lagrangian_cb.episode_rewards))
    smoothed = pd.Series(lagrangian_cb.episode_rewards).rolling(window).mean()
    ax1.plot(smoothed, color='blue', lw=2, label=f'MA{window}')
ax1.axhline(0, color='red', ls='--', alpha=0.5)
ax1.set_title("Episode Reward (CPPO)")
ax1.set_xlabel("Episode")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 에피소드 Cost
ax2 = axes[0, 1]
ax2.plot(lagrangian_cb.episode_costs, alpha=0.5, color='red', lw=0.5)
if len(lagrangian_cb.episode_costs) > 50:
    window = min(50, len(lagrangian_cb.episode_costs))
    smoothed_cost = pd.Series(lagrangian_cb.episode_costs).rolling(window).mean()
    ax2.plot(smoothed_cost, color='red', lw=2, label=f'MA{window}')
ax2.axhline(COST_LIMIT, color='orange', ls='--', lw=2, label=f'limit={COST_LIMIT}')
ax2.set_title("Episode Cost")
ax2.set_xlabel("Episode")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Lambda 변화
ax3 = axes[1, 0]
if lagrangian_cb.lambda_history:
    ax3.plot(lagrangian_cb.lambda_history, color='purple', lw=2, marker='o', markersize=3)
    ax3.set_title(f"λ (Lagrangian Multiplier) — final: {lagrangian_cb.lambda_history[-1]:.4f}")
else:
    ax3.set_title("λ (Lagrangian Multiplier) — no updates yet")
ax3.set_xlabel(f"Update (every {LAMBDA_UPDATE_FREQ} episodes)")
ax3.grid(True, alpha=0.3)

# 4. Cost 유형별 분포
ax4 = axes[1, 1]
if lagrangian_cb.episode_costs_a:
    episodes = range(len(lagrangian_cb.episode_costs_a))
    ax4.fill_between(episodes, lagrangian_cb.episode_costs_a, alpha=0.3, label='A: Drawdown', color='red')
    ax4.fill_between(episodes, lagrangian_cb.episode_costs_b, alpha=0.3, label='B: Transaction', color='orange')
    ax4.fill_between(episodes, lagrangian_cb.episode_costs_c, alpha=0.3, label='C: Hindsight', color='blue')
    ax4.set_title("Cost Breakdown by Type")
    ax4.set_xlabel("Episode")
    ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle("CPPO Training Monitor", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ==========================================
# 10. 테스트 (V1 환경 — 실전에는 cost 제약 없음)
# ==========================================
print(f"\n🧪 테스트 시작... ({TEST_SYMBOL})")
print("   (테스트는 V1 사용 — 실전에서는 미래 데이터/제약 없음)")

test_env = LeverageProEnv([(test_prices, test_techs)], data_stats, leverage=30)
obs, _ = test_env.reset()

results = {'assets': [], 'rewards': [], 'actions': []}
lstm_states = None
episode_start = np.ones((1,), dtype=bool)

while True:
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_start, deterministic=True
    )
    obs, reward, done, _, info = test_env.step(action)
    episode_start = np.array([done])

    results['assets'].append(test_env.total_asset)
    results['rewards'].append(reward)
    results['actions'].append(action[0])

    if done:
        break


# ==========================================
# 11. 결과 시각화
# ==========================================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(results['assets'], color='blue', lw=2)
ax1.axhline(100, color='red', ls='--', alpha=0.5)
ax1.set_title("1. Portfolio Value ($)")
ax1.set_xlabel("Step")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1:])
ax2.plot(test_prices[:len(results['assets'])], color='gray', alpha=0.5)
for t in test_env.trades:
    if t['type'] == 'LONG':
        ax2.scatter(t['step'], t['price'], marker='^', color='green', s=100, zorder=5)
    elif t['type'] == 'SHORT':
        ax2.scatter(t['step'], t['price'], marker='v', color='red', s=100, zorder=5)
    elif t['type'] == 'LIQUIDATION':
        ax2.scatter(t['step'], t['price'], marker='*', color='purple', s=200, zorder=10)
    elif 'EXIT' in t['type']:
        color = 'orange' if t.get('pnl', 0) > 0 else 'brown'
        ax2.scatter(t['step'], t['price'], marker='x', color=color, s=80, zorder=5)
ax2.set_title(f"2. {TEST_SYMBOL} Price & Trade Signals (CPPO)")
ax2.legend(['Price', 'Long', 'Short', 'Liquidation', 'Exit+', 'Exit-'], loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(results['actions'], bins=50, color='teal', alpha=0.7, range=(-1, 1))
ax3.set_title("3. Action Distribution")
ax3.set_xlim(-1.2, 1.2)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(np.cumsum(results['rewards']), color='orange', lw=2)
ax4.axhline(0, color='red', ls='--', alpha=0.5)
ax4.set_title("4. Cumulative Reward (CPPO)")
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
exits = [t for t in test_env.trades if 'pnl_pct' in t]
if exits:
    durations = [t['duration'] for t in exits]
    pnls = [t['pnl_pct'] for t in exits]
    colors = ['green' if p > 0 else 'red' for p in pnls]
    ax5.scatter(durations, pnls, c=colors, alpha=0.6, s=50)
    ax5.axhline(0, color='gray', ls='--', alpha=0.5)
    ax5.set_title("5. Duration vs PnL%")
    ax5.set_xlabel("Duration (steps)")
    ax5.set_ylabel("PnL %")
ax5.grid(True, alpha=0.3)

plt.suptitle("CPPO Test Results", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ==========================================
# 12. 통계 리포트
# ==========================================
print("\n" + "=" * 60)
print(f"📊 CPPO 테스트 결과 리포트 ({TEST_SYMBOL})")
print("=" * 60)

exits = [t for t in test_env.trades if 'pnl' in t]
if exits:
    wins = [t for t in exits if t['pnl'] > 0]
    losses = [t for t in exits if t['pnl'] <= 0]
    liquidations = [t for t in exits if t['type'] == 'LIQUIDATION']

    win_rate = len(wins) / len(exits) * 100 if exits else 0
    avg_pnl = np.mean([t['pnl_pct'] for t in exits])
    total_pnl = sum(t['pnl'] for t in exits)

    print(f"\n💰 최종 자산: ${test_env.total_asset:.2f}")
    print(f"📈 총 수익: ${total_pnl:.2f} ({(test_env.total_asset/100-1)*100:.2f}%)")
    print(f"\n🎯 승률: {win_rate:.1f}% ({len(wins)}승 / {len(losses)}패)")
    print(f"📊 평균 수익률: {avg_pnl:.2f}%")
    print(f"💥 청산 횟수: {len(liquidations)}회")

    if wins:
        avg_win = np.mean([t['pnl_pct'] for t in wins])
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        profit_factor = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else float('inf')
        print(f"\n📏 평균 수익: +{avg_win:.2f}% | 평균 손실: {avg_loss:.2f}%")
        print(f"📏 Profit Factor: {profit_factor:.2f}")

    print(f"\n📋 종료 유형별:")
    for exit_type in ['TP_EXIT', 'TS_EXIT', 'SL_EXIT', 'AI_EXIT', 'LIQUIDATION']:
        count = len([t for t in exits if t['type'] == exit_type])
        if count > 0:
            avg = np.mean([t['pnl_pct'] for t in exits if t['type'] == exit_type])
            print(f"   - {exit_type}: {count}회 (평균 {avg:.2f}%)")

    total_reward = sum(results['rewards'])
    print(f"\n🏆 누적 보상: {total_reward:.3f}")
    print(f"   - 최대: {max(results['rewards']):.3f}")
    print(f"   - 최소: {min(results['rewards']):.3f}")
    print(f"   - 평균: {np.mean(results['rewards']):.4f}")

# CPPO 학습 통계
print(f"\n🛡️ CPPO 학습 통계:")
if lagrangian_cb.lambda_history:
    print(f"   - 최종 λ: {lagrangian_cb.lambda_history[-1]:.4f}")
    print(f"   - 최대 λ: {max(lagrangian_cb.lambda_history):.4f}")
    print(f"   - λ 업데이트 횟수: {len(lagrangian_cb.lambda_history)}")
if lagrangian_cb.episode_costs:
    print(f"   - 평균 에피소드 cost: {np.mean(lagrangian_cb.episode_costs):.4f}")
    print(f"   - 최대 에피소드 cost: {max(lagrangian_cb.episode_costs):.4f}")
    print(f"   - cost_A(드로다운): {np.mean(lagrangian_cb.episode_costs_a):.4f}")
    print(f"   - cost_B(매매손실): {np.mean(lagrangian_cb.episode_costs_b):.4f}")
    print(f"   - cost_C(기회비용): {np.mean(lagrangian_cb.episode_costs_c):.4f}")

print("=" * 60)
