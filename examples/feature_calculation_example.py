"""
새로운 피처 계산 예제 코드
Stock_NeurIPS2018_1_Data.ipynb에 추가할 코드 예시
"""

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

def calculate_advanced_features(df_raw, pair_list, window_size=84):
    """
    새로운 피처들을 계산하는 함수
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        원본 OHLCV 데이터 (MultiIndex: (pair, ['open', 'high', 'low', 'close', 'volume']))
    pair_list : list
        암호화폐 페어 리스트 (예: ['BTC/USDT', 'ETH/USDT'])
    window_size : int
        윈도우 크기 (기본값: 84 = 7시간 * 12개 5분봉)
    
    Returns:
    --------
    dict : 계산된 모든 피처들을 담은 딕셔너리
    """
    results = {}
    
    # 1. OHLCV 배열 추출 (각 코인별로)
    ohlcv_list = []
    price_list = []
    volume_list = []
    
    for pair in pair_list:
        pair_data = df_raw[pair]
        ohlcv = pair_data[['open', 'high', 'low', 'close', 'volume']].values  # shape: (time, 5)
        ohlcv_list.append(ohlcv)
        price_list.append(pair_data['close'].values)
        volume_list.append(pair_data['volume'].values)
    
    # 배열로 변환: (time, crypto_num, 5) for OHLCV, (time, crypto_num) for price/volume
    ohlcv_array = np.stack(ohlcv_list, axis=1)  # shape: (time, crypto_num, 5)
    price_array = np.stack(price_list, axis=1)  # shape: (time, crypto_num)
    volume_array = np.stack(volume_list, axis=1)  # shape: (time, crypto_num)
    
    results['ohlcv_array'] = ohlcv_array
    results['price_array'] = price_array
    
    # 2. EMA 계산 (지수 이동평균)
    ema_list = []
    for pair in pair_list:
        pair_data = df_raw[pair]
        # StockDataFrame으로 변환하여 EMA 계산
        stock_df = Sdf.retype(pair_data.copy())
        ema = stock_df['close_12_ema'].values  # 12기간 EMA
        ema_list.append(ema)
    results['ema_array'] = np.stack(ema_list, axis=1)  # shape: (time, crypto_num)
    
    # 3. 경계 지표 계산
    rsi_list = []
    stochastic_list = []
    bb_percent_b_list = []
    
    for pair in pair_list:
        pair_data = df_raw[pair]
        stock_df = Sdf.retype(pair_data.copy())
        
        # RSI (Relative Strength Index)
        rsi = stock_df['rsi_14'].values  # 14기간 RSI
        rsi_list.append(rsi)
        
        # Stochastic Oscillator (%K)
        stoch_k = stock_df['stoch_k'].values  # Stochastic %K
        stochastic_list.append(stoch_k)
        
        # Bollinger Bands %B
        # %B = (Close - Lower Band) / (Upper Band - Lower Band)
        bb_upper = stock_df['boll_ub'].values
        bb_lower = stock_df['boll_lb'].values
        bb_middle = stock_df['boll'].values
        close = pair_data['close'].values
        bb_percent_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 0으로 나누기 방지
        bb_percent_b_list.append(bb_percent_b)
    
    results['rsi_array'] = np.stack(rsi_list, axis=1)  # shape: (time, crypto_num)
    results['stochastic_array'] = np.stack(stochastic_list, axis=1)  # shape: (time, crypto_num)
    results['bb_percent_b_array'] = np.stack(bb_percent_b_list, axis=1)  # shape: (time, crypto_num)
    
    # 4. 변화율 지표 계산
    price_change_list = []
    volume_change_list = []
    
    for i, pair in enumerate(pair_list):
        # Price Change (%)
        price_change = np.diff(price_array[:, i], prepend=price_array[0, i]) / price_array[:, i] * 100
        price_change_list.append(price_change)
        
        # Volume Change (%)
        volume_change = np.diff(volume_array[:, i], prepend=volume_array[0, i]) / (volume_array[:, i] + 1e-9) * 100
        volume_change_list.append(volume_change)
    
    price_change_array = np.stack(price_change_list, axis=1)  # shape: (time, crypto_num)
    volume_change_array = np.stack(volume_change_list, axis=1)  # shape: (time, crypto_num)
    
    results['price_change_array'] = price_change_array
    results['volume_change_array'] = volume_change_array
    
    # 변화율 지표의 통계 (Z-Score 정규화용)
    results['price_change_mean'] = np.mean(price_change_array, axis=0)  # shape: (crypto_num,)
    results['price_change_std'] = np.std(price_change_array, axis=0)  # shape: (crypto_num,)
    results['volume_change_mean'] = np.mean(volume_change_array, axis=0)  # shape: (crypto_num,)
    results['volume_change_std'] = np.std(volume_change_array, axis=0)  # shape: (crypto_num,)
    
    # 5. 수급/시장 지표 계산
    # OBI (Order Book Imbalance) - 예시: 호가창 불균형 지표
    # 실제로는 호가창 데이터가 필요하지만, 여기서는 가격과 거래량으로 근사치 계산
    obi_list = []
    for i, pair in enumerate(pair_list):
        # 간단한 OBI 근사: (매수 압력 - 매도 압력) / (매수 압력 + 매도 압력)
        # 가격 상승 시 매수 압력, 하락 시 매도 압력으로 가정
        price_diff = np.diff(price_array[:, i], prepend=price_array[0, i])
        volume_norm = volume_array[:, i] / (np.max(volume_array[:, i]) + 1e-9)
        buy_pressure = np.where(price_diff > 0, volume_norm, 0)
        sell_pressure = np.where(price_diff < 0, volume_norm, 0)
        obi = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-9)
        obi_list.append(obi)
    
    results['obi_array'] = np.stack(obi_list, axis=1)  # shape: (time, crypto_num)
    
    # Funding Rate - 실제로는 거래소 API에서 가져와야 함
    # 여기서는 예시로 0으로 초기화 (실제 사용 시 거래소 API에서 가져와야 함)
    funding_rate_array = np.zeros_like(price_array)  # shape: (time, crypto_num)
    results['funding_rate_array'] = funding_rate_array
    
    return results


# ==========================================
# 사용 예제: Stock_NeurIPS2018_1_Data.ipynb에 추가할 코드
# ==========================================

def example_usage():
    """
    Stock_NeurIPS2018_1_Data.ipynb의 데이터 처리 후에 추가할 코드
    """
    
    # 기존 코드에서 df_with_indicators와 CRYPTO_PAIRS가 있다고 가정
    # df_with_indicators = ccxt_eng.add_technical_indicators(...)
    # CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', ...]
    
    # 새로운 피처 계산
    print("새로운 피처 계산 중...")
    feature_dict = calculate_advanced_features(
        df_raw=df_with_indicators,
        pair_list=CRYPTO_PAIRS,
        window_size=84
    )
    
    # 기존 price_array, tech_array와 함께 저장
    price_array, tech_array, date_array = ccxt_eng.df_to_ary(
        df=df_with_indicators,
        pair_list=CRYPTO_PAIRS,
        tech_indicator_list=TECH_INDICATORS
    )
    
    # 학습/테스트 분리
    train_size = int(len(price_array) * 0.8)
    
    train_price = price_array[:train_size]
    train_tech = tech_array[:train_size]
    train_date = date_array[:train_size]
    
    test_price = price_array[train_size:]
    test_tech = tech_array[train_size:]
    test_date = date_array[train_size:]
    
    # 새로운 피처들도 학습/테스트 분리
    train_features = {}
    test_features = {}
    
    for key, value in feature_dict.items():
        if isinstance(value, np.ndarray):
            if len(value.shape) >= 2:  # 시간 차원이 있는 배열
                train_features[key] = value[:train_size]
                test_features[key] = value[train_size:]
            else:  # 통계값 같은 1D 배열
                train_features[key] = value
                test_features[key] = value
    
    # 데이터 저장
    np.savez(
        'crypto_5m_data.npz',
        train_price=train_price,
        train_tech=train_tech,
        train_date=train_date,
        test_price=test_price,
        test_tech=test_tech,
        test_date=test_date,
        crypto_pairs=CRYPTO_PAIRS,
        tech_indicators=TECH_INDICATORS,
        # 새로운 피처들 추가
        **{f'train_{k}': v for k, v in train_features.items()},
        **{f'test_{k}': v for k, v in test_features.items()},
    )
    
    print("데이터 저장 완료!")
    print(f"새로운 피처: {list(feature_dict.keys())}")


# ==========================================
# 환경 생성 시 사용 예제: Stock_NeurIPS2018_2_Train.ipynb에 추가할 코드
# ==========================================

def example_env_creation():
    """
    Stock_NeurIPS2018_2_Train.ipynb의 환경 생성 부분에 추가할 코드
    """
    
    # 데이터 로드
    data = np.load('crypto_5m_data.npz', allow_pickle=True)
    
    train_price = data['train_price']
    train_tech = data['train_tech']
    
    # 새로운 피처들 로드
    train_config = {
        "price_array": train_price,
        "tech_array": train_tech,
        # 새로운 피처들 추가
        "ohlcv_array": data.get('train_ohlcv_array', None),
        "ema_array": data.get('train_ema_array', None),
        "rsi_array": data.get('train_rsi_array', None),
        "stochastic_array": data.get('train_stochastic_array', None),
        "bb_percent_b_array": data.get('train_bb_percent_b_array', None),
        "price_change_array": data.get('train_price_change_array', None),
        "volume_change_array": data.get('train_volume_change_array', None),
        "obi_array": data.get('train_obi_array', None),
        "funding_rate_array": data.get('train_funding_rate_array', None),
        # 통계값들
        "price_change_mean": data.get('price_change_mean', None),
        "price_change_std": data.get('price_change_std', None),
        "volume_change_mean": data.get('volume_change_mean', None),
        "volume_change_std": data.get('volume_change_std', None),
    }
    
    # 환경 생성
    from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
    
    env_train = CryptoEnv(
        config=train_config,
        lookback=1,
        initial_capital=1_000_000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        window_size=84,  # 7시간 (84개 5분봉)
    )
    
    print(f"환경 생성 완료!")
    print(f"State Dim: {env_train.state_dim}")
    print(f"Action Dim: {env_train.action_dim}")
    
    return env_train
