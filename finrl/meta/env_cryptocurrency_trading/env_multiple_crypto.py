from __future__ import annotations

import numpy as np


class CryptoEnv:  # custom env
    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99,
        window_size=84,  # 7시간 (84개 5분봉)
    ):
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.window_size = window_size
        
        # 가격 데이터 (OHLCV + EMA)
        self.price_array = config["price_array"]  # shape: (time, crypto_num)
        self.ohlcv_array = config.get("ohlcv_array", None)  # shape: (time, crypto_num, 5) [open, high, low, close, volume]
        self.ema_array = config.get("ema_array", None)  # shape: (time, crypto_num)
        
        # 기술 지표 배열
        self.tech_array = config["tech_array"]
        
        # 경계 지표 (RSI, Stochastic, BB %B)
        self.rsi_array = config.get("rsi_array", None)
        self.stochastic_array = config.get("stochastic_array", None)
        self.bb_percent_b_array = config.get("bb_percent_b_array", None)
        
        # 변화율 지표 (Price Change, Volume Change)
        self.price_change_array = config.get("price_change_array", None)
        self.volume_change_array = config.get("volume_change_array", None)
        
        # 수급/시장 지표 (OBI, Funding Rate)
        self.obi_array = config.get("obi_array", None)  # Order Book Imbalance
        self.funding_rate_array = config.get("funding_rate_array", None)
        
        # 변화율 지표의 통계 (Z-Score 정규화용)
        self.price_change_mean = config.get("price_change_mean", None)
        self.price_change_std = config.get("price_change_std", None)
        self.volume_change_mean = config.get("volume_change_mean", None)
        self.volume_change_std = config.get("volume_change_std", None)
        
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # reset
        self.time = lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "MulticryptoEnv"
        # 상태 차원 계산: 가격 데이터(5) + 경계 지표(3) + 변화율 지표(2) + 수급/시장 지표(2) = 12개 피처 * crypto_num
        feature_dim = 12 * self.crypto_num  # OHLC + EMA + RSI + Stochastic + BB%B + PriceChange + VolumeChange + OBI + FundingRate
        self.state_dim = feature_dim * lookback
        self.action_dim = self.price_array.shape[1]
        self.if_discrete = False
        self.target_return = 10

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> np.ndarray:
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        state = self.get_state()
        return state

    def step(self, actions) -> (np.ndarray, float, bool, None):
        self.time += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        for index in np.where(actions < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(
                    self.cash // (price[index] * (1 + self.buy_cost_pct)),
                    actions[index],
                )
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * 2**-16
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, None

    def _normalize_price_window(self, prices, window_size=84):
        """윈도우 기반 Min-Max 정규화 (-1 ~ 1)"""
        if len(prices) == 0:
            return np.zeros(prices.shape[1] if len(prices.shape) > 1 else 1)
        
        if len(prices) < window_size:
            window_size = len(prices)
        
        window_prices = prices[-window_size:]
        
        # 2D 배열인 경우
        if len(prices.shape) > 1:
            min_val = np.min(window_prices, axis=0, keepdims=True)
            max_val = np.max(window_prices, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)  # 0으로 나누기 방지
            normalized = 2 * (prices[-1:] - min_val) / range_val - 1
            return normalized.flatten()
        else:
            # 1D 배열인 경우
            min_val = np.min(window_prices)
            max_val = np.max(window_prices)
            range_val = max_val - min_val if max_val != min_val else 1
            normalized = 2 * (prices[-1] - min_val) / range_val - 1
            return np.array([normalized])
    
    def _normalize_boundary_indicator(self, indicator_value, center=0, scale=1.0):
        """경계 지표 선형 변환 (Center at 0)"""
        # RSI: 0-100 -> -1 to 1 (50이 중립)
        # Stochastic: 0-100 -> -1 to 1 (50이 중립)
        # BB %B: 보통 0-1 -> -1 to 1 (0.5가 중립)
        if indicator_value is None or np.isnan(indicator_value):
            return 0.0
        if center == 0:
            # BB %B 같은 경우: 0-1 범위를 -1~1로 변환 (0.5가 중립)
            return (indicator_value - 0.5) * 2.0
        else:
            # RSI, Stochastic 같은 경우: 0-100 범위를 -1~1로 변환 (center가 중립)
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

    def get_state(self):
        """
        관측값 생성 (계좌/포지션 상태 제외)
        - 가격 데이터: Open, High, Low, Close, EMA (윈도우 기반 Min-Max -1~1)
        - 경계 지표: RSI, Stochastic, BB %B (선형 변환, Center at 0)
        - 변화율 지표: Price Change, Volume Change (Z-Score & Clipping)
        - 수급/시장 지표: OBI, Funding Rate (Raw Scaling -1~1)
        """
        state_list = []
        
        for i in range(self.lookback):
            t = self.time - i
            if t < 0:
                t = 0
            
            for crypto_idx in range(self.crypto_num):
                # 1. 가격 데이터 (윈도우 기반 Min-Max -1~1)
                if self.ohlcv_array is not None and t < len(self.ohlcv_array):
                    # Open, High, Low, Close
                    ohlc = self.ohlcv_array[max(0, t-self.window_size+1):t+1, crypto_idx, :4]  # shape: (window, 4)
                    if len(ohlc) > 0:
                        ohlc_normalized = self._normalize_price_window(ohlc, self.window_size)
                    else:
                        ohlc_normalized = np.zeros(4)
                else:
                    # OHLC 데이터가 없으면 Close 가격만 사용
                    close_prices = self.price_array[max(0, t-self.window_size+1):t+1, crypto_idx]
                    ohlc_normalized = np.repeat(self._normalize_price_window(close_prices.reshape(-1, 1), self.window_size), 4)
                
                # EMA
                if self.ema_array is not None and t < len(self.ema_array):
                    ema_prices = self.ema_array[max(0, t-self.window_size+1):t+1, crypto_idx]
                    ema_normalized = self._normalize_price_window(ema_prices.reshape(-1, 1), self.window_size)
                else:
                    ema_normalized = np.array([0.0])
                
                state_list.extend(ohlc_normalized[:4])  # Open, High, Low, Close
                state_list.append(ema_normalized[0])  # EMA
                
                # 2. 경계 지표 (선형 변환, Center at 0)
                rsi_val = self.rsi_array[t, crypto_idx] if (self.rsi_array is not None and t < len(self.rsi_array) and crypto_idx < self.rsi_array.shape[1]) else None
                rsi_norm = self._normalize_boundary_indicator(rsi_val, center=50.0)
                
                stoch_val = self.stochastic_array[t, crypto_idx] if (self.stochastic_array is not None and t < len(self.stochastic_array) and crypto_idx < self.stochastic_array.shape[1]) else None
                stoch_norm = self._normalize_boundary_indicator(stoch_val, center=50.0)
                
                bb_val = self.bb_percent_b_array[t, crypto_idx] if (self.bb_percent_b_array is not None and t < len(self.bb_percent_b_array) and crypto_idx < self.bb_percent_b_array.shape[1]) else None
                bb_norm = self._normalize_boundary_indicator(bb_val, center=0.5)
                
                state_list.extend([rsi_norm, stoch_norm, bb_norm])
                
                # 3. 변화율 지표 (Z-Score & Clipping)
                price_change_val = self.price_change_array[t, crypto_idx] if (self.price_change_array is not None and t < len(self.price_change_array) and crypto_idx < self.price_change_array.shape[1]) else None
                price_change_mean = self.price_change_mean[crypto_idx] if (self.price_change_mean is not None and crypto_idx < len(self.price_change_mean)) else None
                price_change_std = self.price_change_std[crypto_idx] if (self.price_change_std is not None and crypto_idx < len(self.price_change_std)) else None
                price_change_norm = self._normalize_change_indicator(price_change_val, price_change_mean, price_change_std)
                
                volume_change_val = self.volume_change_array[t, crypto_idx] if (self.volume_change_array is not None and t < len(self.volume_change_array) and crypto_idx < self.volume_change_array.shape[1]) else None
                volume_change_mean = self.volume_change_mean[crypto_idx] if (self.volume_change_mean is not None and crypto_idx < len(self.volume_change_mean)) else None
                volume_change_std = self.volume_change_std[crypto_idx] if (self.volume_change_std is not None and crypto_idx < len(self.volume_change_std)) else None
                volume_change_norm = self._normalize_change_indicator(volume_change_val, volume_change_mean, volume_change_std)
                
                state_list.extend([price_change_norm, volume_change_norm])
                
                # 4. 수급/시장 지표 (Raw Scaling -1~1)
                obi_val = self.obi_array[t, crypto_idx] if (self.obi_array is not None and t < len(self.obi_array) and crypto_idx < self.obi_array.shape[1]) else None
                obi_norm = self._normalize_market_indicator(obi_val)
                
                funding_val = self.funding_rate_array[t, crypto_idx] if (self.funding_rate_array is not None and t < len(self.funding_rate_array) and crypto_idx < self.funding_rate_array.shape[1]) else None
                funding_norm = self._normalize_market_indicator(funding_val)
                
                state_list.extend([obi_norm, funding_norm])
        
        state = np.array(state_list, dtype=np.float32)
        return state

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = len(str(price)) - 7
            action_norm_vector.append(1 / ((10) ** x))

        self.action_norm_vector = np.asarray(action_norm_vector)
