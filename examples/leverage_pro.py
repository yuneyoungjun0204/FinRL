"""
================================================================================
Leverage Pro - 30x Isolated Margin Scalping RL Environment
================================================================================
보상 함수 설계 원칙:
1. Equity-Change Reward: reward = delta(equity) / initial_equity
   → 실제 자산 변화와 보상이 직결되어 Reward Hacking 원천 차단
2. VecNormalize: 자동 보상 스케일링으로 Value Function 안정화
3. 거래 쿨다운: 연속 매매 방지 (최소 5스텝 간격)
4. 청산 추가 페널티: 자산 손실분 + -0.5 추가 억제
5. 손절(SL) 추가 페널티: -0.15 (손절선까지 끌려가는 행위 억제)
6. 관망 보상: +0.0002/스텝 (확실하지 않으면 쉬는 게 이득)
7. 에피소드 조기 종료: -20% 드로다운 시 즉시 truncation
8. 에피소드 생존 보너스: ROI 비례 (최대 ±0.3)

청산 조건: Isolated Margin 30x에서 약 3.33% 역행 시 청산

데이터: Multi-Symbol (8종목) × Multi-Timeframe (5m + 1h)
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
# 1. 학습 모니터링 콜백
# ==========================================
class RewardMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

# ==========================================
# 2. 데이터 수집 (Multi-Symbol, Multi-Timeframe)
# ==========================================
TRAIN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"
]
TEST_SYMBOL = "ADAUSDT"

# ==========================================
# 모델 아키텍처 설정 ("MLP" 또는 "LSTM")
# ==========================================
MODEL_TYPE = "LSTM"

# ==========================================
# 커리큘럼 학습 설정 (ML-Agents --initialize-from 방식)
# ==========================================
# RUN_ID: 현재 학습의 고유 ID (모델 저장 경로)
#   예: "Stage1_Basic", "Stage2_Advanced"
RUN_ID = "Stage1_Basic"

# INITIALIZE_FROM: 이전 학습 모델 경로 (None이면 처음부터 학습)
#   예: "Stage1_Basic" → runs/Stage1_Basic/ 에서 모델 로드 후 이어서 학습
#   ML-Agents의 --initialize-from=Stage1_Basic 과 동일
INITIALIZE_FROM = None

# 저장 경로
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SAVE_DIR = os.path.join(RUNS_DIR, RUN_ID)

# 체크포인트 저장 간격 (timesteps 단위)
CHECKPOINT_FREQ = 100000


def fetch_raw_ohlcv(exchange, symbol, timeframe, total_limit):
    """Raw OHLCV 데이터 수집"""
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
    """기술적 지표 계산 및 상대적 피처로 변환"""
    p = prefix

    df[f'{p}returns'] = df['close'].pct_change()
    ma7 = df['close'].rolling(7).mean()
    ma25 = df['close'].rolling(25).mean()
    df[f'{p}volatility'] = df[f'{p}returns'].rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df[f'{p}rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df[f'{p}bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # 상대적 피처로 변환 (절대 가격 스케일 제거)
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
    """한 종목의 5m + 1h 멀티 타임프레임 데이터 수집"""
    print(f"   📡 {symbol}", end=" ")

    # 5분봉
    df_5m = fetch_raw_ohlcv(exchange, symbol, "5m", total_limit)
    df_5m, feat_5m = compute_indicators(df_5m, prefix='')

    # 1시간봉 (5m 범위를 커버 + rolling window 여유)
    h1_limit = total_limit // 12 + 200
    df_1h = fetch_raw_ohlcv(exchange, symbol, "1h", h1_limit)
    df_1h, feat_1h = compute_indicators(df_1h, prefix='h1_')

    # 1h 데이터 정렬 준비 (Look-ahead 방지: 완성된 캔들만 사용)
    # 1h 캔들 timestamp는 시작 시각 → +3600000ms = 종료 시각
    # merge_asof backward: 5m 시점 이전에 완성된 1h 캔들만 매칭
    df_1h_merge = df_1h[['timestamp'] + feat_1h].copy()
    df_1h_merge['ts_1h_close'] = df_1h_merge['timestamp'] + 3600000
    df_1h_merge = df_1h_merge.drop(columns=['timestamp'])

    # 5m ← 1h 정렬
    df_merged = pd.merge_asof(
        df_5m.sort_values('timestamp'),
        df_1h_merge.sort_values('ts_1h_close'),
        left_on='timestamp',
        right_on='ts_1h_close',
        direction='backward'
    )

    # 전체 피처 컬럼: 5m(9) + 1h(9) = 18
    all_feat_cols = feat_5m + feat_1h

    # NaN 제거 (rolling window 미충족 행)
    nan_mask = df_merged[all_feat_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    df_clean = df_merged[~nan_mask].reset_index(drop=True)

    price_array = df_clean[['close']].values
    tech_array = df_clean[all_feat_cols].values

    print(f"→ 5m:{len(df_5m)} + 1h:{len(df_1h)} → {len(df_clean)}개 (NaN제거:{nan_count}) [피처:{tech_array.shape[1]}]")
    return price_array, tech_array, all_feat_cols


# 데이터 로드
print("=" * 60)
print("📊 Multi-Symbol, Multi-Timeframe 데이터 수집")
print(f"   학습 종목: {', '.join(TRAIN_SYMBOLS)}")
print(f"   테스트 종목: {TEST_SYMBOL}")
print("=" * 60)

exchange = ccxt.bybit()
all_symbol_data = {}

for sym in sorted(set(TRAIN_SYMBOLS + [TEST_SYMBOL])):
    all_symbol_data[sym] = fetch_symbol_data(exchange, sym, total_limit=60000)
    time.sleep(1)  # API rate limit 방지

# 학습 데이터 구성 (각 종목 앞 80%)
train_datasets = []
all_train_techs = []

for sym in TRAIN_SYMBOLS:
    prices, techs, feat_cols = all_symbol_data[sym]
    n = int(len(prices) * 0.8)
    train_datasets.append((prices[:n], techs[:n]))
    all_train_techs.append(techs[:n])

# 테스트 데이터 (TEST_SYMBOL 뒤 20%)
test_prices_full, test_techs_full, _ = all_symbol_data[TEST_SYMBOL]
test_split = int(len(test_prices_full) * 0.8)
test_prices = test_prices_full[test_split:]
test_techs = test_techs_full[test_split:]

# 정규화 통계 (모든 학습 데이터 기준, Look-ahead bias 방지)
all_train_combined = np.concatenate(all_train_techs, axis=0)
data_stats = {
    'tech_mean': all_train_combined.mean(axis=0),
    'tech_std': all_train_combined.std(axis=0) + 1e-9
}

total_train_samples = sum(len(t) for _, t in train_datasets)
print(f"\n   ✅ 학습: {len(TRAIN_SYMBOLS)}종목 × 80% = {total_train_samples:,}개")
print(f"   ✅ 테스트: {TEST_SYMBOL} 20% = {len(test_prices):,}개")
print(f"   ✅ 피처: {all_train_combined.shape[1]}개 (5m:9 + 1h:9)")
print(f"   ✅ 정규화: 학습 데이터 기준")

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 3. Equity-Change Reward (Reward Hacking 원천 차단)
# ==========================================
class EquityChangeRewardCalculator:
    """
    Equity-Change Reward: reward = delta(equity) / initial_equity

    설계 원칙:
    1. 보상 = 실제 자산 변화 → Reward Hacking 원천 차단
       (자산이 줄면 보상도 반드시 줄어듦)
    2. tanh 압축 없음 → 큰 손실 = 비례하는 큰 음수 보상
       (기존: -100% 손실이나 -10% 손실이나 비슷한 보상)
    3. 수수료가 자산을 깎음 → 과매매 시 자연적 보상 감소
    4. VecNormalize가 자동 스케일링 담당
    """

    def __init__(self, leverage=30):
        self.leverage = leverage
        self.initial_equity = 100.0
        self.prev_equity = 100.0

    def reset(self, initial_equity):
        self.initial_equity = initial_equity
        self.prev_equity = initial_equity

    def calculate_reward(self, curr_equity, is_liquidated=False,
                         is_sl_exit=False, has_position=True,
                         is_episode_end=False, survived=True):
        """
        보상 = (현재 자산 - 이전 자산) / 초기 자산

        추가 신호:
        - 청산: -0.5 추가 페널티
        - 손절(SL): -0.15 추가 페널티 (손절선까지 가지 않도록 학습)
        - 관망: +0.0002 (확실하지 않으면 쉬는 게 이득)
        - 에피소드 생존: ROI 비례 보너스 (최대 ±0.3)
        """
        # 핵심: 자산 변화율 (실제 PnL과 직결)
        equity_change = (curr_equity - self.prev_equity) / self.initial_equity
        reward = equity_change

        # 청산 추가 페널티 (자산 손실분 + 추가 억제)
        if is_liquidated:
            reward -= 0.5

        # 손절 추가 페널티 (SL선까지 끌려가는 행위 억제)
        if is_sl_exit:
            reward -= 0.15

        # 관망 보상 ("확실하지 않으면 쉬는 게 이득")
        if not has_position:
            reward += 0.0002

        # 에피소드 생존 보너스 (전체 성과 평가)
        if is_episode_end and survived:
            roi = (curr_equity - self.initial_equity) / self.initial_equity
            reward += np.clip(roi, -0.3, 0.3)

        self.prev_equity = curr_equity
        return reward


# ==========================================
# 4. 강화학습 환경 (Multi-Dataset 지원)
# ==========================================
class LeverageProEnv(gym.Env):
    """
    30x Isolated Margin Scalping Environment

    Multi-Dataset: reset() 시 랜덤 종목 선택으로 학습 다양성 확보
    State: 정규화된 기술적 지표 (5m:9 + 1h:9 = 18) + 포지션 상태 (4)
    Action: [-1, 1] (음수=숏, 양수=롱, 0근처=관망)
    Reward: Equity-Change (자산 변화 기반) + VecNormalize 자동 스케일링
    """

    def __init__(self, datasets, stats,
                 leverage=35, initial_capital=100.0, entry_size=5.0):
        super().__init__()

        # datasets = [(price_array, tech_array), ...]
        self.datasets = datasets
        self.stats = stats

        self.leverage = leverage
        self.initial_capital = initial_capital
        self.entry_size = entry_size
        self.taker_fee = 0.00055

        # 청산 임계값 (30x: 약 3.33% 역행)
        self.liquidation_pct = 1.0 / leverage

        # 손절/익절 설정
        self.stop_loss_pct = 0.02      # 2% 손절
        self.take_profit_pct = 0.03    # 3% 익절
        self.trailing_activation = 0.02  # 2% 수익 시 트레일링 활성화
        self.trailing_callback = 0.005   # 0.5% 콜백

        # 과매매 방지: 거래 간 최소 쿨다운 (5스텝 = 25분)
        self.trade_cooldown = 5

        # 보상 계산기 (Equity-Change)
        self.reward_calc = EquityChangeRewardCalculator(leverage=leverage)

        # 첫 번째 데이터셋으로 shape 결정
        self.price_array, self.tech_array = self.datasets[0]

        # 상태 공간: 기술적 지표(18) + 포지션 상태(4)
        self.state_dim = self.tech_array.shape[1] + 4
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.max_step = len(self.price_array) - 2
        self.reset()

    def _normalize_tech(self, tech):
        """기술적 지표 Z-Score 정규화"""
        normalized = (tech - self.stats['tech_mean']) / self.stats['tech_std']
        return np.clip(normalized, -3, 3)  # 이상치 클리핑

    def _get_obs(self):
        """상태 관측값 생성"""
        # 기술적 지표 (정규화)
        tech = self._normalize_tech(self.tech_array[self.time])

        # 포지션 상태
        position_side = np.sign(self.position) if self.position != 0 else 0
        position_pnl = self.unrealized_pnl / self.entry_size if self.position != 0 else 0
        holding_time = (self.time - self.entry_time) / 100 if self.position != 0 else 0
        cash_ratio = self.cash / self.initial_capital

        state = np.concatenate([
            tech,
            [position_side, position_pnl, holding_time, cash_ratio]
        ])

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        # 랜덤 데이터셋 선택 (에피소드마다 다른 종목)
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
        """포지션 청산 및 PnL 계산"""
        curr_price = self.price_array[self.time][0]
        fee = abs(self.position) * self.taker_fee
        realized_pnl = self.unrealized_pnl - fee

        self.cash += self.entry_size + realized_pnl
        pnl_pct = (realized_pnl / self.entry_size) * 100
        duration = self.time - self.entry_time

        self.trades.append({
            'step': self.time,
            'type': reason,
            'price': curr_price,
            'pnl': realized_pnl,
            'pnl_pct': pnl_pct,
            'duration': duration
        })

        self.position = 0
        self.entry_price = 0
        self.trailing_active = False
        self.best_price_pct = 0

        return pnl_pct, duration

    def _open_position(self, action):
        """포지션 진입"""
        curr_price = self.price_array[self.time][0]
        pos_value = self.entry_size * self.leverage
        fee = pos_value * self.taker_fee

        self.cash -= (self.entry_size + fee)
        self.position = pos_value if action > 0 else -pos_value
        self.entry_price = curr_price
        self.entry_time = self.time

        self.trades.append({
            'step': self.time,
            'type': 'LONG' if action > 0 else 'SHORT',
            'price': curr_price
        })

    def step(self, actions):
        self.time += 1
        curr_price = self.price_array[self.time][0]
        action = float(actions[0])

        pnl_pct = 0
        is_liquidated = False
        is_sl_exit = False

        # 포지션 관리
        if self.position != 0:
            # 미실현 손익 계산
            price_change = (curr_price - self.entry_price) / self.entry_price
            if self.position < 0:
                price_change = -price_change

            self.unrealized_pnl = abs(self.position) * price_change

            # 청산 체크 (Isolated Margin)
            if price_change <= -self.liquidation_pct:
                pnl_pct, _ = self._close_position("LIQUIDATION")
                is_liquidated = True
            # 손절
            elif price_change <= -self.stop_loss_pct:
                pnl_pct, _ = self._close_position("SL_EXIT")
                is_sl_exit = True
            # 익절
            elif price_change >= self.take_profit_pct:
                pnl_pct, _ = self._close_position("TP_EXIT")
            # 트레일링 스탑
            elif self.trailing_active:
                if price_change > self.best_price_pct:
                    self.best_price_pct = price_change
                elif self.best_price_pct - price_change >= self.trailing_callback:
                    pnl_pct, _ = self._close_position("TS_EXIT")
            else:
                # 트레일링 활성화 체크
                if price_change >= self.trailing_activation:
                    self.trailing_active = True
                    self.best_price_pct = price_change

                # AI 판단 청산
                if abs(action) < 0.3 or (action * self.position < 0):
                    pnl_pct, _ = self._close_position("AI_EXIT")

        # 신규 진입 (쿨다운 적용: 연속 매매 방지)
        line = 0.8
        cooldown_ok = (self.time - self.last_trade_time) >= self.trade_cooldown
        if self.position == 0 and abs(action) > line and self.cash >= self.entry_size and cooldown_ok:
            self._open_position(action)
            self.last_trade_time = self.time

        # 총 자산 업데이트
        self.total_asset = self.cash + (self.entry_size if self.position != 0 else 0) + self.unrealized_pnl

        # 드로다운 체크 및 에피소드 종료 판정 (-20% 이상 손실 시 즉시 종료)
        is_alive = self.total_asset >= self.initial_capital * 0.40
        is_episode_end = self.time >= self.max_step
        done = is_episode_end or not is_alive

        # Equity-Change Reward 계산
        reward = self.reward_calc.calculate_reward(
            curr_equity=self.total_asset,
            is_liquidated=is_liquidated,
            is_sl_exit=is_sl_exit,
            has_position=(self.position != 0),
            is_episode_end=is_episode_end,
            survived=(is_alive and is_episode_end)
        )

        self.prev_action = action

        return self._get_obs(), reward, done, False, {
            'total_asset': self.total_asset,
            'pnl_pct': pnl_pct
        }


# ==========================================
# 5. 모델 아키텍처 모듈
# ==========================================

# --- MLP 설정 ---
# 22 → 256 → LeakyReLU → 128 → LeakyReLU → 1
MLP_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    activation_fn=th.nn.LeakyReLU,
    log_std_init=-0.55,
)

# --- LSTM 설정 ---
# 22 → LSTM(128) → 128 → LeakyReLU → 1
# shared_lstm=False: Actor/Critic 각각 독립 LSTM
# enable_critic_lstm=True: Critic도 LSTM으로 시계열 가치 추정
LSTM_POLICY_KWARGS = dict(
    lstm_hidden_size=128,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
    net_arch=dict(pi=[128], vf=[128]),
    activation_fn=th.nn.LeakyReLU,
    log_std_init=-0.55,
)

# --- PPO 공통 하이퍼파라미터 ---
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
    """
    모델 생성 또는 이전 학습에서 로드 (커리큘럼 학습 지원)

    Args:
        env: VecNormalize로 래핑된 학습 환경
        model_type: "MLP" 또는 "LSTM"
        initialize_from: 이전 RUN_ID (None이면 새로 생성)
            예: "Stage1_Basic" → runs/Stage1_Basic/best_model.zip 로드

    Returns:
        PPO 또는 RecurrentPPO 모델
    """
    # --- initialize-from: 이전 모델에서 가중치 로드 ---
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

        # VecNormalize 통계도 로드 (있으면)
        prev_vecnorm_path = os.path.join(prev_dir, "vecnormalize.pkl")
        if os.path.exists(prev_vecnorm_path):
            env = VecNormalize.load(prev_vecnorm_path, env.venv)
            model.set_env(env)
            print(f"   📂 VecNormalize 통계 로드 완료")

        return model

    # --- 새 모델 생성 ---
    if model_type == "LSTM":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            policy_kwargs=LSTM_POLICY_KWARGS,
            learning_rate=1e-4,
            **PPO_HYPERPARAMS
        )
        print(f"\n🧠 모델: RecurrentPPO (LSTM) — 새로 생성")
        print(f"   Input(22) → LSTM(128) → Dense(128) → LeakyReLU → Action(1)")
        print(f"   Actor LSTM + Critic LSTM (독립)")
    else:
        model = PPO(
            "MlpPolicy", env,
            policy_kwargs=MLP_POLICY_KWARGS,
            learning_rate=2e-4,
            **PPO_HYPERPARAMS
        )
        print(f"\n🧠 모델: PPO (MLP) — 새로 생성")
        print(f"   Input(22) → Dense(256) → LeakyReLU → Dense(128) → LeakyReLU → Action(1)")

    return model


# ==========================================
# 6. 학습 실행
# ==========================================
print("\n" + "=" * 60)
print("🚀 Leverage Pro - 30x Isolated Margin Scalping")
print("=" * 60)
print(f"📊 학습: {len(TRAIN_SYMBOLS)}종목 ({', '.join(TRAIN_SYMBOLS)})")
print(f"📊 테스트: {TEST_SYMBOL}")
print(f"📈 학습 데이터: {total_train_samples:,}개 / 테스트: {len(test_prices):,}개")
print(f"📐 피처: 5m(9) + 1h(9) = 18 + 포지션(4) = 22차원 상태")
print(f"🧠 모델 아키텍처: {MODEL_TYPE}")
print(f"⚡ 레버리지: 30x (Isolated)")
print(f"🎯 청산 임계값: {100/30:.2f}% 역행")
print(f"💾 RUN_ID: {RUN_ID}" + (f" (initialize-from: {INITIALIZE_FROM})" if INITIALIZE_FROM else ""))
print("\n💡 보상 함수 구성 (Equity-Change + VecNormalize):")
print("   - 매 스텝: delta(equity) / initial_equity — 실제 자산 변화 직결")
print("   - 청산: 자산 손실 + 추가 -0.5 페널티")
print("   - 손절(SL): 자산 손실 + 추가 -0.15 페널티")
print("   - 관망 보상: +0.0002/스텝 (쉬는 게 이득)")
print("   - 과매매 방지: 수수료 자연 억제 + 5스텝 쿨다운")
print("   - 에피소드 조기 종료: -20% 드로다운 시 truncation")
print("   - 에피소드 생존: ROI 비례 보너스 (최대 ±0.3)")
print("   - VecNormalize: 자동 보상 스케일링")
print("=" * 60)

# 환경 생성 (Multi-Dataset + VecNormalize)
# DummyVecEnv로 래핑 후 VecNormalize 적용
# norm_obs=False: 관측값은 자체 Z-Score 정규화 사용
# norm_reward=True: 보상을 자동으로 running std로 나눠 Value Function 안정화
# clip_reward=10.0: 정규화 후 안전 클리핑
# gamma=0.98: PPO gamma와 동일하게 설정
train_env = DummyVecEnv([lambda: LeverageProEnv(train_datasets, data_stats, leverage=30)])
train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                         clip_reward=10.0, gamma=0.98)

# 모델 생성 또는 이전 모델에서 로드
model = create_model(train_env, MODEL_TYPE, initialize_from=INITIALIZE_FROM)
total_timesteps_num = 1000000

# 저장 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

# 콜백: 모니터링 + 체크포인트 (중단해도 모델 보존)
callback = RewardMonitorCallback()
checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=os.path.join(SAVE_DIR, "checkpoints"),
    name_prefix=RUN_ID,
    save_vecnormalize=True,
)

# 학습
init_msg = f" (initialize-from: {INITIALIZE_FROM})" if INITIALIZE_FROM else " (새로 시작)"
print(f"\n🏋️ 학습 시작...{init_msg}")
print(f"   💾 저장 경로: {SAVE_DIR}")
print(f"   💾 체크포인트: {CHECKPOINT_FREQ:,} 스텝마다 자동 저장")
model.learn(total_timesteps=total_timesteps_num, callback=[callback, checkpoint_cb])

# 학습 완료 후 최종 모델 저장
model.save(os.path.join(SAVE_DIR, "best_model"))
train_env.save(os.path.join(SAVE_DIR, "vecnormalize.pkl"))
print(f"\n💾 최종 모델 저장 완료:")
print(f"   📂 {os.path.join(SAVE_DIR, 'best_model.zip')}")
print(f"   📂 {os.path.join(SAVE_DIR, 'vecnormalize.pkl')}")

# 학습 곡선
plt.figure(figsize=(12, 4))
plt.plot(callback.episode_rewards, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title("Training Reward per Episode (Equity-Change + VecNormalize)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 6. 테스트
# ==========================================
print(f"\n🧪 테스트 시작... ({TEST_SYMBOL})")
test_env = LeverageProEnv([(test_prices, test_techs)], data_stats, leverage=30)
obs, _ = test_env.reset()

results = {'assets': [], 'rewards': [], 'actions': []}

# LSTM/MLP 공용 predict 루프
# MLP는 state/episode_start를 무시하고 None 반환
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
# 7. 결과 시각화
# ==========================================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3)

# 1. 자산 추이
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(results['assets'], color='blue', lw=2)
ax1.axhline(100, color='red', ls='--', alpha=0.5)
ax1.set_title("1. Portfolio Value ($)")
ax1.set_xlabel("Step")
ax1.grid(True, alpha=0.3)

# 2. 가격 및 거래 신호
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

ax2.set_title(f"2. {TEST_SYMBOL} Price & Trade Signals")
ax2.legend(['Price', 'Long', 'Short', 'Liquidation', 'Exit+', 'Exit-'], loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. 액션 분포
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(results['actions'], bins=50, color='teal', alpha=0.7, range=(-1, 1))
ax3.set_title("3. Action Distribution")
ax3.set_xlim(-1.2, 1.2)
ax3.grid(True, alpha=0.3)

# 4. 누적 보상 (정규화됨)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(np.cumsum(results['rewards']), color='orange', lw=2)
ax4.axhline(0, color='red', ls='--', alpha=0.5)
ax4.set_title("4. Cumulative Reward (Equity-Change)")
ax4.grid(True, alpha=0.3)

# 5. 거래 효율성
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

plt.tight_layout()
plt.show()

# ==========================================
# 8. 통계 리포트
# ==========================================
print("\n" + "=" * 60)
print(f"📊 테스트 결과 리포트 ({TEST_SYMBOL})")
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

    print(f"\n📋 종료 유형별:")
    for exit_type in ['TP_EXIT', 'TS_EXIT', 'SL_EXIT', 'AI_EXIT', 'LIQUIDATION']:
        count = len([t for t in exits if t['type'] == exit_type])
        if count > 0:
            avg = np.mean([t['pnl_pct'] for t in exits if t['type'] == exit_type])
            print(f"   - {exit_type}: {count}회 (평균 {avg:.2f}%)")

    # 누적 보상 통계
    total_reward = sum(results['rewards'])
    print(f"\n🏆 누적 보상: {total_reward:.3f} (Equity-Change, 원본값)")
    print(f"   - 최대: {max(results['rewards']):.3f}")
    print(f"   - 최소: {min(results['rewards']):.3f}")
    print(f"   - 평균: {np.mean(results['rewards']):.4f}")

print("=" * 60)
