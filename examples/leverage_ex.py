import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stockstats import StockDataFrame as Sdf
import torch as th  # 정책망 활성화 함수용
import os
import platform

# ==========================================
# 1. 학습 로그 수집용 콜백 (Training Monitor)
# ==========================================
class TrainingVisualizerCallback(BaseCallback):
    """학습 도중 에피소드별 보상을 기록하는 콜백"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.curr_reward = 0

    def _on_step(self) -> bool:
        self.curr_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self.curr_reward)
            self.curr_reward = 0
        return True

# ==========================================
# 2. 데이터 수집 및 분할 (7000개 수집)
# ==========================================
TARGET_SYMBOL = "ADAUSDT" 

def calculate_advanced_features(df, window_size=84):
    """
    새로운 피처들을 계산하는 함수
    
    Returns:
    --------
    dict : 계산된 모든 피처들을 담은 딕셔너리
    """
    results = {}
    
    # OHLCV 배열
    ohlcv_array = df[['open', 'high', 'low', 'close', 'volume']].values  # shape: (time, 5)
    price_array = df['close'].values.reshape(-1, 1)  # shape: (time, 1)
    volume_array = df['volume'].values.reshape(-1, 1)  # shape: (time, 1)
    
    results['ohlcv_array'] = ohlcv_array.reshape(-1, 1, 5)  # shape: (time, crypto_num=1, 5)
    results['price_array'] = price_array
    
    # EMA 계산
    stock_df = Sdf.retype(df.copy())
    try:
        ema = stock_df['close_12_ema'].values
        ema = np.nan_to_num(ema, nan=0.0).reshape(-1, 1)
    except:
        # EMA가 없으면 SMA로 대체
        ema = df['close'].rolling(window=12).mean().values
        ema = np.nan_to_num(ema, nan=0.0).reshape(-1, 1)
    results['ema_array'] = ema
    
    # 경계 지표 계산 (RSI, Stochastic, BB %B)
    # RSI
    try:
        rsi = stock_df['rsi_14'].values
        rsi = np.nan_to_num(rsi, nan=50.0).reshape(-1, 1)  # NaN이면 중립값 50
    except:
        # RSI 수동 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        rsi = np.nan_to_num(rsi, nan=50.0).reshape(-1, 1)
    results['rsi_array'] = rsi
    
    # Stochastic
    try:
        stoch_k = stock_df['stoch_k'].values
        stoch_k = np.nan_to_num(stoch_k, nan=50.0).reshape(-1, 1)
    except:
        # Stochastic 수동 계산
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        stoch_k = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-9)
        stoch_k = np.nan_to_num(stoch_k, nan=50.0).reshape(-1, 1)
    results['stochastic_array'] = stoch_k
    
    # Bollinger Bands %B
    try:
        bb_upper = stock_df['boll_ub'].values
        bb_lower = stock_df['boll_lb'].values
    except:
        # BB 수동 계산
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
    
    close = df['close'].values
    bb_percent_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    bb_percent_b = np.nan_to_num(bb_percent_b, nan=0.5).reshape(-1, 1)  # NaN이면 중립값 0.5
    results['bb_percent_b_array'] = bb_percent_b
    
    # 변화율 지표 계산
    price_change = np.diff(price_array[:, 0], prepend=price_array[0, 0]) / (price_array[:, 0] + 1e-9) * 100
    volume_change = np.diff(volume_array[:, 0], prepend=volume_array[0, 0]) / (volume_array[:, 0] + 1e-9) * 100
    
    price_change = np.nan_to_num(price_change, nan=0.0)
    volume_change = np.nan_to_num(volume_change, nan=0.0)
    
    price_change_array = price_change.reshape(-1, 1)
    volume_change_array = volume_change.reshape(-1, 1)
    
    results['price_change_array'] = price_change_array
    results['volume_change_array'] = volume_change_array
    
    # 변화율 지표의 통계 (Z-Score 정규화용)
    results['price_change_mean'] = np.mean(price_change_array, axis=0)
    results['price_change_std'] = np.std(price_change_array, axis=0)
    results['volume_change_mean'] = np.mean(volume_change_array, axis=0)
    results['volume_change_std'] = np.std(volume_change_array, axis=0)
    
    # 수급/시장 지표 계산
    # OBI (Order Book Imbalance) - 가격/거래량으로 근사
    price_diff = np.diff(price_array[:, 0], prepend=price_array[0, 0])
    volume_norm = volume_array[:, 0] / (np.max(volume_array[:, 0]) + 1e-9)
    buy_pressure = np.where(price_diff > 0, volume_norm, 0)
    sell_pressure = np.where(price_diff < 0, volume_norm, 0)
    obi = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-9)
    results['obi_array'] = obi.reshape(-1, 1)
    
    # Funding Rate - 실제로는 거래소 API에서 가져와야 함 (여기서는 0으로 초기화)
    results['funding_rate_array'] = np.zeros_like(price_array)
    
    return results

def fetch_bybit_multi_limit(symbol="BTCUSDT", timeframe="5m", total_limit=10000):
    print(f"📡 Bybit {symbol} {total_limit}개 데이터 수집 중...")
    exchange = ccxt.bybit()
    all_ohlcv = []
    last_ts = None
    
    while len(all_ohlcv) < total_limit:
        limit = min(1000, total_limit - len(all_ohlcv))
        params = {}
        if last_ts: params['endTime'] = last_ts - 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        if not ohlcv: break
        all_ohlcv = ohlcv + all_ohlcv
        last_ts = ohlcv[0][0]
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 새로운 피처 계산
    print("새로운 피처 계산 중...")
    feature_dict = calculate_advanced_features(df, window_size=84)
    
    # 기존 호환성을 위한 price_array, tech_array (사용하지 않음)
    price_array = df[['close']].values
    tech_array = np.zeros((len(df), 43))  # 더미 배열
    
    return price_array, tech_array, feature_dict

# 데이터 수집
total_prices, total_techs, total_features = fetch_bybit_multi_limit(TARGET_SYMBOL, total_limit=10000)

# 학습/테스트 분리
train_size = 8000
train_prices = total_prices[:train_size]
train_techs = total_techs[:train_size]
test_prices = total_prices[train_size:]
test_techs = total_techs[train_size:]

# 피처들도 학습/테스트 분리
train_features = {}
test_features = {}
for key, value in total_features.items():
    if isinstance(value, np.ndarray):
        if len(value.shape) >= 2:  # 시간 차원이 있는 배열
            train_features[key] = value[:train_size]
            test_features[key] = value[train_size:]
        else:  # 통계값 같은 1D 배열
            train_features[key] = value
            test_features[key] = value

if platform.system() == 'Windows': plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 3. 강화학습 환경 (개선된 보상 함수 적용)
# ==========================================
class CryptoAnslEnv(gym.Env):
    def __init__(self, price_array, tech_array, feature_dict=None, leverage=35, entry_size=5.0, window_size=84):
        super().__init__()
        self.price_array = price_array
        self.tech_array = tech_array  # tech_array 저장
        self.leverage, self.entry_size = leverage, entry_size
        self.initial_capital, self.taker_fee = 100.0, 0.0004
        self.stop_loss_pct, self.ts_activation_pct, self.ts_callback_pct = 0.01, 0.02, 0.0002
        self.window_size = window_size
        
        self.action_penalty_coef = 0.01  # 액션 페널티 (0.5 → 0.01, 보상 왜곡 방지)
        self.entry_penalty_val = 0.1
        self.time_penalty = 0.001  # 매 스텝 타임 페널티 (1,000스텝 = -1.0)
        self.bankruptcy_penalty = -1.0  # 파산 페널티 ([-1, 1] 범위에 맞춤: -5.0 → -1.0)        
        
        # 새로운 피처들 로드
        if feature_dict is None:
            feature_dict = {}
        
        self.ohlcv_array = feature_dict.get("ohlcv_array", None)
        self.ema_array = feature_dict.get("ema_array", None)
        self.rsi_array = feature_dict.get("rsi_array", None)
        self.stochastic_array = feature_dict.get("stochastic_array", None)
        self.bb_percent_b_array = feature_dict.get("bb_percent_b_array", None)
        self.price_change_array = feature_dict.get("price_change_array", None)
        self.volume_change_array = feature_dict.get("volume_change_array", None)
        self.obi_array = feature_dict.get("obi_array", None)
        self.funding_rate_array = feature_dict.get("funding_rate_array", None)
        self.price_change_mean = feature_dict.get("price_change_mean", None)
        self.price_change_std = feature_dict.get("price_change_std", None)
        self.volume_change_mean = feature_dict.get("volume_change_mean", None)
        self.volume_change_std = feature_dict.get("volume_change_std", None)
        
        # 기본 상태 차원: 12개 피처 (OHLC + EMA + RSI + Stochastic + BB%B + PriceChange + VolumeChange + OBI + FundingRate)
        base_state_dim = 12
        
        # tech_array 차원 추가
        tech_dim = 0
        if self.tech_array is not None:
            try:
                if isinstance(self.tech_array, np.ndarray):
                    if len(self.tech_array.shape) == 1:
                        tech_dim = self.tech_array.shape[0]
                    elif len(self.tech_array.shape) == 2:
                        tech_dim = self.tech_array.shape[1]  # (time, features)
                    elif len(self.tech_array.shape) > 2:
                        tech_dim = np.prod(self.tech_array.shape[1:])  # 첫 번째 차원 제외한 모든 차원의 곱
                elif len(self.tech_array) > 0:
                    # 리스트나 다른 형태인 경우
                    sample = self.tech_array[0] if hasattr(self.tech_array, '__getitem__') else self.tech_array
                    if isinstance(sample, np.ndarray):
                        tech_dim = sample.flatten().shape[0]
                    else:
                        tech_dim = 1
            except:
                tech_dim = 0
        
        # 상태 차원 계산 (기본 피처 + tech_array)
        self.state_dim = base_state_dim + tech_dim
        
        # tech_array StandardScaler 스타일 정규화를 위한 통계값 계산
        if self.tech_array is not None:
            try:
                if isinstance(self.tech_array, np.ndarray):
                    if len(self.tech_array.shape) == 2:
                        # (time, features) 형태인 경우
                        self.tech_mean = np.mean(self.tech_array, axis=0)
                        self.tech_std = np.std(self.tech_array, axis=0)
                    else:
                        # 1D 또는 다른 형태
                        self.tech_mean = np.mean(self.tech_array)
                        self.tech_std = np.std(self.tech_array)
                else:
                    self.tech_mean = None
                    self.tech_std = None
            except:
                self.tech_mean = None
                self.tech_std = None
        else:
            self.tech_mean = None
            self.tech_std = None
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.max_step = len(self.price_array) - 2
        self.reset()

    def _normalize_price_window(self, prices, window_size=84):
        """윈도우 기반 Min-Max 정규화 (-1 ~ 1)"""
        if len(prices) == 0:
            return np.zeros(prices.shape[1] if len(prices.shape) > 1 else 1)
        
        if len(prices) < window_size:
            window_size = len(prices)
        
        window_prices = prices[-window_size:]
        
        if len(prices.shape) > 1:
            min_val = np.min(window_prices, axis=0, keepdims=True)
            max_val = np.max(window_prices, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = 2 * (prices[-1:] - min_val) / range_val - 1
            return normalized.flatten()
        else:
            min_val = np.min(window_prices)
            max_val = np.max(window_prices)
            range_val = max_val - min_val if max_val != min_val else 1
            normalized = 2 * (prices[-1] - min_val) / range_val - 1
            return np.array([normalized])
    
    def _normalize_boundary_indicator(self, indicator_value, center=0):
        """경계 지표 선형 변환 (Center at 0)"""
        if indicator_value is None or np.isnan(indicator_value):
            return 0.0
        if center == 0:
            return (indicator_value - 0.5) * 2.0
        else:
            return (indicator_value - center) / center
    
    def _normalize_change_indicator(self, change_value, mean_val, std_val, clip_range=3.0):
        """변화율 지표 Z-Score & Clipping"""
        if change_value is None or np.isnan(change_value) or mean_val is None or std_val is None or std_val == 0:
            return 0.0
        z_score = (change_value - mean_val) / std_val
        return np.clip(z_score, -clip_range, clip_range)
    
    def _normalize_market_indicator(self, indicator_value):
        """수급/시장 지표 Raw Scaling (-1 ~ 1 유지)"""
        if indicator_value is None or np.isnan(indicator_value):
            return 0.0
        return np.clip(indicator_value, -1.0, 1.0)

    def _get_obs(self):
        """
        새로운 관측값 생성 (계좌/포지션 상태 제외)
        - 가격 데이터: Open, High, Low, Close, EMA (윈도우 기반 Min-Max -1~1)
        - 경계 지표: RSI, Stochastic, BB %B (선형 변환, Center at 0)
        - 변화율 지표: Price Change, Volume Change (Z-Score & Clipping)
        - 수급/시장 지표: OBI, Funding Rate (Raw Scaling -1~1)
        """
        t = self.time
        crypto_idx = 0  # 단일 코인
        
        state_list = []
        
        # 1. 가격 데이터 (윈도우 기반 Min-Max -1~1)
        start_idx = max(0, t - self.window_size + 1)
        end_idx = t + 1
        
        if self.ohlcv_array is not None and t < len(self.ohlcv_array):
            ohlc = self.ohlcv_array[start_idx:end_idx, crypto_idx, :4]
            if len(ohlc) > 0:
                ohlc_normalized = self._normalize_price_window(ohlc, min(self.window_size, len(ohlc)))
            else:
                ohlc_normalized = np.zeros(4)
        else:
            close_prices = self.price_array[start_idx:end_idx, 0]
            if len(close_prices) > 0:
                ohlc_normalized = np.repeat(self._normalize_price_window(close_prices.reshape(-1, 1), min(self.window_size, len(close_prices))), 4)
            else:
                ohlc_normalized = np.zeros(4)
        
        # EMA
        if self.ema_array is not None and t < len(self.ema_array):
            ema_prices = self.ema_array[start_idx:end_idx, crypto_idx]
            if len(ema_prices) > 0:
                ema_normalized = self._normalize_price_window(ema_prices.reshape(-1, 1), min(self.window_size, len(ema_prices)))
            else:
                ema_normalized = np.array([0.0])
        else:
            ema_normalized = np.array([0.0])
        
        state_list.extend(ohlc_normalized[:4])  # Open, High, Low, Close
        state_list.append(ema_normalized[0])  # EMA
        
        # 2. 경계 지표 (선형 변환, Center at 0)
        rsi_val = self.rsi_array[t, crypto_idx] if (self.rsi_array is not None and t < len(self.rsi_array)) else None
        rsi_norm = self._normalize_boundary_indicator(rsi_val, center=50.0)
        
        stoch_val = self.stochastic_array[t, crypto_idx] if (self.stochastic_array is not None and t < len(self.stochastic_array)) else None
        stoch_norm = self._normalize_boundary_indicator(stoch_val, center=50.0)
        
        bb_val = self.bb_percent_b_array[t, crypto_idx] if (self.bb_percent_b_array is not None and t < len(self.bb_percent_b_array)) else None
        bb_norm = self._normalize_boundary_indicator(bb_val, center=0.5)
        
        state_list.extend([rsi_norm, stoch_norm, bb_norm])
        
        # 3. 변화율 지표 (Z-Score & Clipping)
        price_change_val = self.price_change_array[t, crypto_idx] if (self.price_change_array is not None and t < len(self.price_change_array)) else None
        price_change_mean = self.price_change_mean[crypto_idx] if (self.price_change_mean is not None and crypto_idx < len(self.price_change_mean)) else None
        price_change_std = self.price_change_std[crypto_idx] if (self.price_change_std is not None and crypto_idx < len(self.price_change_std)) else None
        price_change_norm = self._normalize_change_indicator(price_change_val, price_change_mean, price_change_std)
        
        volume_change_val = self.volume_change_array[t, crypto_idx] if (self.volume_change_array is not None and t < len(self.volume_change_array)) else None
        volume_change_mean = self.volume_change_mean[crypto_idx] if (self.volume_change_mean is not None and crypto_idx < len(self.volume_change_mean)) else None
        volume_change_std = self.volume_change_std[crypto_idx] if (self.volume_change_std is not None and crypto_idx < len(self.volume_change_std)) else None
        volume_change_norm = self._normalize_change_indicator(volume_change_val, volume_change_mean, volume_change_std)
        
        state_list.extend([price_change_norm, volume_change_norm])
        
        # 4. 수급/시장 지표 (Raw Scaling -1~1)
        obi_val = self.obi_array[t, crypto_idx] if (self.obi_array is not None and t < len(self.obi_array)) else None
        obi_norm = self._normalize_market_indicator(obi_val)
        
        funding_val = self.funding_rate_array[t, crypto_idx] if (self.funding_rate_array is not None and t < len(self.funding_rate_array)) else None
        funding_norm = self._normalize_market_indicator(funding_val)
        
        state_list.extend([obi_norm, funding_norm])
        
        # 5. tech_array StandardScaler 스타일 정규화 (입력 데이터 단순화)
        if self.tech_array is not None and t < len(self.tech_array):
            tech_val = self.tech_array[t]
            if isinstance(tech_val, np.ndarray):
                tech_val = tech_val.flatten()
                # StandardScaler 스타일: (x - mean) / std
                if self.tech_mean is not None and self.tech_std is not None:
                    tech_mean_flat = self.tech_mean.flatten() if isinstance(self.tech_mean, np.ndarray) else self.tech_mean
                    tech_std_flat = self.tech_std.flatten() if isinstance(self.tech_std, np.ndarray) else self.tech_std
                    
                    # 차원이 맞지 않으면 브로드캐스팅 시도
                    if tech_mean_flat.shape == tech_val.shape:
                        tech_normalized = (tech_val - tech_mean_flat) / (tech_std_flat + 1e-9)
                    else:
                        # 스칼라인 경우
                        tech_normalized = (tech_val - tech_mean_flat) / (tech_std_flat + 1e-9)
                    state_list.extend(tech_normalized.tolist())
                else:
                    # 통계값이 없으면 원본 사용 (초기화 시 계산 실패)
                    state_list.extend(tech_val.tolist())
            else:
                state_list.append(float(tech_val))
        
        state = np.array(state_list, dtype=np.float32)
        return state

    def reset(self, seed=None, options=None):
        self.time, self.cash = 0, self.initial_capital
        self.positions, self.entry_prices, self.unrealized_pnl = np.zeros(1), np.zeros(1), np.zeros(1)
        self.total_asset, self.best_price, self.ts_activated = self.cash, 0.0, False
        self.trades = []
        self.entry_time = 0
        self.prev_action = 0.0
        
        # 실제 관측값 생성하여 차원 확인 및 observation_space 업데이트
        obs = self._get_obs()
        actual_dim = obs.shape[0]
        if actual_dim != self.state_dim:
            self.state_dim = actual_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        return obs, {}

    def _close_position(self, curr_price, reason="EXIT"):
        """
        포지션 종료 및 수익률 비례 확정 보상 계산 ([-1, 1] 범위로 정규화)
        Returns: (reward, pnl_pct) - 정규화된 보상과 수익률 퍼센트
        """
        fee = abs(self.positions[0]) * self.taker_fee
        realized_pnl = self.unrealized_pnl[0] - fee
        duration = self.time - self.entry_time
        self.cash += (self.entry_size + realized_pnl)
        
        # 소수점 수익률 계산 (예: 5% → 0.05)
        pnl_decimal = realized_pnl / self.entry_size
        pnl_pct = pnl_decimal * 100  # 로깅용 퍼센트
        
        # 확정 수익 보상 스케일링: [-1, 1] 범위로 정규화
        # 수익률 50%일 때 최대 보상 1.0, -50%일 때 최소 페널티 -1.0
        reward = np.clip(pnl_decimal * 2.0, -1.0, 1.0)
        
        self.trades.append({'step': self.time, 'type': reason, 'price': curr_price, 
                            'pnl': realized_pnl, 'duration': duration, 'pnl_pct': pnl_pct})
        
        # 로깅: 매매 종료 시 pnl_pct와 reward 출력
        print(f"  [거래 종료] {reason:8s} | 수익률: {pnl_pct:7.2f}% | 보상: {reward:6.3f} (정규화)")
        
        self.positions[0], self.ts_activated, self.best_price = 0, False, 0.0
        return reward, pnl_pct

    def step(self, actions):
        """
        수익률 비례 확정 보상 체계 적용 ([-1, 1] 범위로 정규화):
        - 미실현 보상 제거 (포지션 유지 중 자산 변화 보상 없음)
        - 포지션 종료 시에만 수익률 비례 보상 (pnl_decimal * 2.0, 클리핑 [-1, 1])
        - 매 스텝 타임 페널티 (-0.001)
        - 파산 페널티 (-1.0)
        - 최종 보상 클리핑으로 발산 물리적 차단
        """
        self.time += 1
        curr_price = self.price_array[self.time][0]
        action_val = float(actions[0])
        
        # 기본 페널티 초기화
        r_penalty = 0.0
        r_realized = 0.0  # 확정 보상 (포지션 종료 시에만)
        
        # 1. 액션 페널티 (상향 조정: 0.5)
        r_penalty -= self.action_penalty_coef * (action_val ** 2)
        
        # 2. 타임 페널티 (매 스텝 -0.02)
        r_penalty -= self.time_penalty

        # 3. 포지션 관리
        if self.positions[0] != 0:
            # 미실현 손익 업데이트 (보상에는 반영하지 않음)
            side_change = (curr_price - self.entry_prices[0]) / self.entry_prices[0]
            if self.positions[0] < 0: side_change = -side_change
            self.unrealized_pnl[0] = abs(self.positions[0]) * side_change
            
            # Isolated Margin 청산 체크 (-100% 손실 시 청산) - 최우선 체크
            if side_change <= -1.0:  # -100% 손실 = 청산
                r_realized, _ = self._close_position(curr_price, "LIQUIDATION")
            # Trailing Stop 체크
            elif self.ts_activated and (self.best_price - side_change) >= self.ts_callback_pct:
                r_realized, _ = self._close_position(curr_price, "TS_EXIT")
            elif side_change <= -self.stop_loss_pct:
                r_realized, _ = self._close_position(curr_price, "SL_EXIT")
            elif abs(actions[0]) < 0.5 or (actions[0] * self.positions[0] < 0):
                r_realized, _ = self._close_position(curr_price, "EXIT")
            else:
                # 포지션 유지 중: Trailing Stop 활성화 체크
                if side_change > self.best_price: self.best_price = side_change
                if side_change >= self.ts_activation_pct: self.ts_activated = True

        # 4. 포지션 진입
        if abs(actions[0]) > 0.5 and self.positions[0] == 0 and self.cash >= self.entry_size:
            r_penalty -= self.entry_penalty_val
            pos_value = self.entry_size * self.leverage
            self.cash -= (self.entry_size + pos_value * self.taker_fee)
            self.positions[0], self.entry_prices[0] = (pos_value if actions[0] > 0 else -pos_value), curr_price
            self.entry_time = self.time
            self.trades.append({'step': self.time, 'type': 'LONG' if actions[0] > 0 else 'SHORT', 'price': curr_price})

        # 5. 총 자산 업데이트
        self.total_asset = self.cash + (self.entry_size if self.positions[0] != 0 else 0) + self.unrealized_pnl[0]
        
        # 6. 파산 페널티 체크
        is_bankrupt = self.total_asset < 10.0
        if is_bankrupt:
            r_penalty += self.bankruptcy_penalty
            print(f"  [파산] 총 자산: ${self.total_asset:.2f} | 페널티: {self.bankruptcy_penalty}")
        
        # 7. 최종 보상 계산 (미실현 보상 제거, 확정 보상만 사용)
        final_reward = r_realized + r_penalty
        
        # 보상값을 [-1, 1] 범위로 물리적으로 클리핑하여 발산 방지
        final_reward = np.clip(final_reward, -1.0, 1.0)
        
        self.prev_action = action_val

        done = self.time >= self.max_step or is_bankrupt
        return self._get_obs(), final_reward, done, False, {'realized': r_realized, 'penalty': r_penalty}

# ==========================================
# 4. 학습 시작 및 과정 시각화 (Train Phase)
# ==========================================
print("🏋️ [Train Phase] 학습 시작 (8000 Step 구간)...")
print("📊 보상값 [-1, 1] 범위 정규화 체계 적용")
print("   - 확정 수익 보상: pnl_decimal * 2.0 → clip([-1, 1])")
print("   - 매 스텝 타임 페널티: -0.001 (1,000스텝 = -1.0)")
print("   - 파산 페널티: -1.0 (정규화 범위 내)")
print("   - 액션 페널티 계수: 0.01 (보상 왜곡 방지)")
print("   - 정책망: [256, 128] (Tanh 활성화)")
print("   - 학습률: 0.0001 (1e-4, 미세한 차이 학습 가능)")
print("   - 엔트로피 계수: 0.1 (탐색 효율 유지)")
print("   - 최종 보상 클리핑: np.clip([-1, 1])로 발산 물리적 차단\n")

monitor_callback = TrainingVisualizerCallback()

# 정책망 아키텍처 및 활성화 함수 설정
policy_kwargs = dict(
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    activation_fn=th.nn.Tanh,  # Tanh 활성화 (음수 신호 유지 및 포화 억제)
)

model = PPO(
    "MlpPolicy", 
    CryptoAnslEnv(train_prices, train_techs, train_features), 
    policy_kwargs=policy_kwargs,
    learning_rate=0.0001,  # 보상 스케일이 작아졌으므로 미세한 차이 학습 가능 (1e-4)
    ent_coef=0.1,  # 탐색 효율 유지
    verbose=1
).learn(total_timesteps=300000, callback=monitor_callback)

# --- 학습 로그 Plot ---
plt.figure(figsize=(12, 6))
plt.plot(monitor_callback.ep_rewards, color='green', alpha=0.7)
plt.title("Training Reward per Episode (Learning Progress)", fontsize=14, fontweight='bold')
plt.xlabel("Episode Count")
plt.ylabel("Total Reward")
plt.grid(True, alpha=0.3)
print("\n💡 학습 과정 차트를 닫으면 실전 테스트로 넘어갑니다.")
plt.show()

# ==========================================
# 5. 실전 테스트 및 5단 통합 리포트 (Test Phase)
# ==========================================
print("\n🚀 [Test Phase] 최신 2000개 데이터로 테스트 진행...")
test_env = CryptoAnslEnv(test_prices, test_techs, test_features)
obs, _ = test_env.reset()
diag = {'assets': [], 'actions': [], 'reward_total': []}

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    diag['assets'].append(test_env.total_asset)
    diag['actions'].append(action[0])
    diag['reward_total'].append(reward)
    if done: break

# --- 통합 분석 리포트 ---
fig = plt.figure(figsize=(24, 15))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
x_axis = np.arange(len(diag['assets']))

# 1. 자산 변동
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_axis, diag['assets'], color='#1f77b4', lw=2)
ax1.axhline(100, color='red', ls='--')
ax1.set_title("1. Test Asset Trend (USD)", fontweight='bold')

# 2. 가격 및 상세 타점
ax2 = fig.add_subplot(gs[0, 1:])
ax2.plot(x_axis, test_prices[:len(diag['assets'])], color='gray', alpha=0.3)
ax2.scatter([], [], color='#2ca02c', marker='^', s=100, label='▲ LONG Entry')
ax2.scatter([], [], color='#d62728', marker='v', s=100, label='▼ SHORT Entry')
ax2.scatter([], [], color='#ff7f0e', marker='x', s=100, label='✕ TS EXIT')
ax2.scatter([], [], color='#9467bd', marker='x', s=100, label='✕ SL EXIT')
ax2.scatter([], [], color='#e74c3c', marker='*', s=150, label='★ LIQUIDATION')
for t in test_env.trades:
    c = '#2ca02c' if t['type'] == 'LONG' else '#d62728' if t['type'] == 'SHORT' else 'black'
    if 'EXIT' in t['type'] or t['type'] == 'LIQUIDATION':
        if t['type'] == 'TS_EXIT':
            c = '#ff7f0e'
        elif t['type'] == 'SL_EXIT':
            c = '#9467bd'
        elif t['type'] == 'LIQUIDATION':
            c = '#e74c3c'
    m = '^' if t['type'] == 'LONG' else 'v' if t['type'] == 'SHORT' else '*' if t['type'] == 'LIQUIDATION' else 'x'
    ax2.scatter(t['step'], t['price'], color=c, marker=m, s=150 if m == '*' else 100, 
                edgecolors='white' if m not in ['x', '*'] else 'black', linewidths=2 if m == '*' else 1, zorder=10)
ax2.legend(loc='upper right', title="Trade Legend"); ax2.set_title("2. Execution Signals")

# 3. Action 분포 (짤림 방지)
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(diag['actions'], bins=60, range=(-1.1, 1.1), color='teal', alpha=0.7)
ax3.set_xlim([-1.2, 1.2]); ax3.set_title("3. Action Confidence Distribution")

# 4. 누적 보상
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x_axis, np.cumsum(diag['reward_total']), color='#ff7f0e', lw=2)
ax4.set_title("4. Cumulative Improved Reward")

# 5. 매매 효율성
ax5 = fig.add_subplot(gs[1, 2])
exits = [t for t in test_env.trades if 'EXIT' in t['type']]
if exits:
    pnls = [t['pnl_pct'] for t in exits]
    sc = ax5.scatter([t['duration'] for t in exits], pnls, c=pnls, cmap='RdYlGn', s=100, edgecolors='black')
    plt.colorbar(sc, ax=ax5); ax5.set_title("5. Duration vs PnL %")

plt.tight_layout()
print("\n💡 테스트 결과 차트를 닫으면 최종 승률 보고서가 출력됩니다.")
plt.show()

# 최종 통계 출력
ex_list = [t for t in test_env.trades if 'EXIT' in t['type']]
if ex_list:
    wr = len([t for t in ex_list if t['pnl'] > 0]) / len(ex_list) * 100
    
    # 종료 유형별 횟수 계산
    ts_exits = [t for t in ex_list if t['type'] == 'TS_EXIT']  # 익절 (Trailing Stop)
    sl_exits = [t for t in ex_list if t['type'] == 'SL_EXIT']  # 손절 (Stop Loss)
    ai_exits = [t for t in ex_list if t['type'] == 'EXIT']     # 자진 포지션 정리 (AI 판단)
    liquidation_exits = [t for t in ex_list if t['type'] == 'LIQUIDATION']  # 청산 (-100% 손실)
    liquidation_count = len(liquidation_exits)
    
    print(f"\n📊 실전 테스트 결과 | 승률: {wr:.2f}% | 총 매매: {len(ex_list)}회")
    print(f"   ├─ 익절 (Trailing Stop): {len(ts_exits)}회")
    print(f"   ├─ 손절 (Stop Loss): {len(sl_exits)}회")
    print(f"   ├─ 자진 포지션 정리 (AI 판단): {len(ai_exits)}회")
    print(f"   └─ 청산 (Liquidation): {liquidation_count}회")
    
    # 종료 유형별 상세 분석
    print("\n" + "="*70)
    print("🔍 포지션 종료 유형별 상세 분석")
    print("="*70)
    
    # 1. 청산 (Liquidation) - Isolated Margin에서 -100% 손실 시 청산
    print(f"\n💥 청산 (Liquidation): {liquidation_count}회")
    if liquidation_count > 0:
        avg_liquidation_pnl = np.mean([t['pnl_pct'] for t in liquidation_exits])
        print(f"   → 평균 손익률: {avg_liquidation_pnl:.2f}% (예상: -100%)")
        print(f"   → 최종 자산: ${test_env.total_asset:.2f}")
    
    # 2. 손절 (Stop Loss)
    sl_exits = [t for t in ex_list if t['type'] == 'SL_EXIT']
    sl_win = len([t for t in sl_exits if t['pnl'] > 0])
    sl_loss = len([t for t in sl_exits if t['pnl'] <= 0])
    print(f"\n🛑 손절 (Stop Loss): {len(sl_exits)}회")
    if len(sl_exits) > 0:
        print(f"   → 승리: {sl_win}회 | 손실: {sl_loss}회")
        avg_sl_pnl = np.mean([t['pnl_pct'] for t in sl_exits])
        print(f"   → 평균 손익률: {avg_sl_pnl:.2f}%")
    
    # 3. 익절 (Trailing Stop)
    ts_exits = [t for t in ex_list if t['type'] == 'TS_EXIT']
    ts_win = len([t for t in ts_exits if t['pnl'] > 0])
    ts_loss = len([t for t in ts_exits if t['pnl'] <= 0])
    print(f"\n📈 익절 (Trailing Stop): {len(ts_exits)}회")
    if len(ts_exits) > 0:
        print(f"   → 승리: {ts_win}회 | 손실: {ts_loss}회")
        avg_ts_pnl = np.mean([t['pnl_pct'] for t in ts_exits])
        print(f"   → 평균 손익률: {avg_ts_pnl:.2f}%")
    
    # 4. AI 판단 종료 (일반 EXIT)
    ai_exits = [t for t in ex_list if t['type'] == 'EXIT']
    ai_win = len([t for t in ai_exits if t['pnl'] > 0])
    ai_loss = len([t for t in ai_exits if t['pnl'] <= 0])
    print(f"\n🤖 AI 판단 종료 (EXIT): {len(ai_exits)}회")
    if len(ai_exits) > 0:
        print(f"   → 승리: {ai_win}회 | 손실: {ai_loss}회")
        avg_ai_pnl = np.mean([t['pnl_pct'] for t in ai_exits])
        print(f"   → 평균 손익률: {avg_ai_pnl:.2f}%")
    
    # 종합 요약
    print("\n" + "="*70)
    print("📋 종합 요약")
    print("="*70)
    total_exits = len(ex_list)
    print(f"총 종료 횟수: {total_exits}회")
    print(f"  - 청산: {liquidation_count}회 ({liquidation_count/total_exits*100:.2f}%)")
    print(f"  - 손절: {len(sl_exits)}회 ({len(sl_exits)/total_exits*100:.2f}%)")
    print(f"  - 익절: {len(ts_exits)}회 ({len(ts_exits)/total_exits*100:.2f}%)")
    print(f"  - AI 판단: {len(ai_exits)}회 ({len(ai_exits)/total_exits*100:.2f}%)")
    
    # 승률 분석
    print(f"\n승률 분석:")
    total_wins = len([t for t in ex_list if t['pnl'] > 0])
    total_losses = len([t for t in ex_list if t['pnl'] <= 0])
    print(f"  - 전체 승리: {total_wins}회 ({total_wins/total_exits*100:.2f}%)")
    print(f"  - 전체 손실: {total_losses}회 ({total_losses/total_exits*100:.2f}%)")
    
    # 평균 손익률
    if ex_list:
        avg_pnl = np.mean([t['pnl_pct'] for t in ex_list])
        print(f"\n평균 손익률: {avg_pnl:.2f}%")
        
        # 최고/최악 거래
        best_trade = max(ex_list, key=lambda x: x['pnl_pct'])
        worst_trade = min(ex_list, key=lambda x: x['pnl_pct'])
        print(f"\n최고 거래: {best_trade['pnl_pct']:.2f}% ({best_trade['type']})")
        print(f"최악 거래: {worst_trade['pnl_pct']:.2f}% ({worst_trade['type']})")
    
    print("="*70)