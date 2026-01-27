# 새로운 피처 추가 가이드

## 개요
`env_multiple_crypto.py`를 수정하여 새로운 피처들을 받을 수 있도록 했습니다. 
이제 **데이터 전처리 단계**에서 이러한 피처들을 계산하고, **환경 생성 시** config에 추가해야 합니다.

---

## 1단계: 데이터 전처리 (Stock_NeurIPS2018_1_Data.ipynb)

### 기존 코드 위치
`Stock_NeurIPS2018_1_Data.ipynb`의 "Part 4: 데이터 배열 변환 및 저장" 섹션

### 추가할 코드

#### 1-1. 새로운 피처 계산 함수 추가

```python
import numpy as np
from stockstats import StockDataFrame as Sdf

def calculate_advanced_features(df_with_indicators, pair_list, window_size=84):
    """
    새로운 피처들을 계산하는 함수
    
    Returns:
    --------
    dict : 계산된 모든 피처들을 담은 딕셔너리
    """
    results = {}
    
    # 1. OHLCV 배열 추출
    ohlcv_list = []
    price_list = []
    volume_list = []
    
    for pair in pair_list:
        pair_data = df_with_indicators[pair]
        ohlcv = pair_data[['open', 'high', 'low', 'close', 'volume']].values
        ohlcv_list.append(ohlcv)
        price_list.append(pair_data['close'].values)
        volume_list.append(pair_data['volume'].values)
    
    ohlcv_array = np.stack(ohlcv_list, axis=1)  # shape: (time, crypto_num, 5)
    price_array = np.stack(price_list, axis=1)   # shape: (time, crypto_num)
    volume_array = np.stack(volume_list, axis=1) # shape: (time, crypto_num)
    
    results['ohlcv_array'] = ohlcv_array
    results['price_array'] = price_array
    
    # 2. EMA 계산
    ema_list = []
    for pair in pair_list:
        pair_data = df_with_indicators[pair]
        stock_df = Sdf.retype(pair_data.copy())
        ema = stock_df['close_12_ema'].values  # 12기간 EMA
        ema_list.append(ema)
    results['ema_array'] = np.stack(ema_list, axis=1)
    
    # 3. 경계 지표 계산 (RSI, Stochastic, BB %B)
    rsi_list = []
    stochastic_list = []
    bb_percent_b_list = []
    
    for pair in pair_list:
        pair_data = df_with_indicators[pair]
        stock_df = Sdf.retype(pair_data.copy())
        
        # RSI
        rsi = stock_df['rsi_14'].values
        rsi_list.append(rsi)
        
        # Stochastic
        stoch_k = stock_df['stoch_k'].values
        stochastic_list.append(stoch_k)
        
        # Bollinger Bands %B
        bb_upper = stock_df['boll_ub'].values
        bb_lower = stock_df['boll_lb'].values
        close = pair_data['close'].values
        bb_percent_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
        bb_percent_b_list.append(bb_percent_b)
    
    results['rsi_array'] = np.stack(rsi_list, axis=1)
    results['stochastic_array'] = np.stack(stochastic_list, axis=1)
    results['bb_percent_b_array'] = np.stack(bb_percent_b_list, axis=1)
    
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
    
    price_change_array = np.stack(price_change_list, axis=1)
    volume_change_array = np.stack(volume_change_list, axis=1)
    
    results['price_change_array'] = price_change_array
    results['volume_change_array'] = volume_change_array
    
    # 변화율 지표의 통계 (Z-Score 정규화용)
    results['price_change_mean'] = np.mean(price_change_array, axis=0)
    results['price_change_std'] = np.std(price_change_array, axis=0)
    results['volume_change_mean'] = np.mean(volume_change_array, axis=0)
    results['volume_change_std'] = np.std(volume_change_array, axis=0)
    
    # 5. 수급/시장 지표 계산
    # OBI (Order Book Imbalance) - 가격/거래량으로 근사
    obi_list = []
    for i, pair in enumerate(pair_list):
        price_diff = np.diff(price_array[:, i], prepend=price_array[0, i])
        volume_norm = volume_array[:, i] / (np.max(volume_array[:, i]) + 1e-9)
        buy_pressure = np.where(price_diff > 0, volume_norm, 0)
        sell_pressure = np.where(price_diff < 0, volume_norm, 0)
        obi = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-9)
        obi_list.append(obi)
    
    results['obi_array'] = np.stack(obi_list, axis=1)
    
    # Funding Rate - 실제로는 거래소 API에서 가져와야 함 (여기서는 0으로 초기화)
    results['funding_rate_array'] = np.zeros_like(price_array)
    
    return results
```

#### 1-2. 기존 데이터 처리 코드 수정

**기존 코드:**
```python
# 배열로 변환
price_array, tech_array, date_array = ccxt_eng.df_to_ary(
    df=df_with_indicators,
    pair_list=CRYPTO_PAIRS,
    tech_indicator_list=TECH_INDICATORS
)

# 학습/테스트 분리
train_size = int(len(price_array) * 0.8)
train_price = price_array[:train_size]
train_tech = tech_array[:train_size]
# ...
```

**수정된 코드 (추가 부분):**
```python
# 배열로 변환
price_array, tech_array, date_array = ccxt_eng.df_to_ary(
    df=df_with_indicators,
    pair_list=CRYPTO_PAIRS,
    tech_indicator_list=TECH_INDICATORS
)

# ===== 여기에 추가 =====
# 새로운 피처 계산
print("새로운 피처 계산 중...")
feature_dict = calculate_advanced_features(
    df_with_indicators=df_with_indicators,
    pair_list=CRYPTO_PAIRS,
    window_size=84
)
# =======================

# 학습/테스트 분리
train_size = int(len(price_array) * 0.8)
train_price = price_array[:train_size]
train_tech = tech_array[:train_size]
train_date = date_array[:train_size]

test_price = price_array[train_size:]
test_tech = tech_array[train_size:]
test_date = date_array[train_size:]

# ===== 여기에 추가 =====
# 새로운 피처들도 학습/테스트 분리
train_features = {}
test_features = {}

for key, value in feature_dict.items():
    if isinstance(value, np.ndarray):
        if len(value.shape) >= 2:  # 시간 차원이 있는 배열
            train_features[key] = value[:train_size]
            test_features[key] = value[train_size:]
        else:  # 통계값 같은 1D 배열 (mean, std)
            train_features[key] = value
            test_features[key] = value
# =======================
```

#### 1-3. 데이터 저장 코드 수정

**기존 코드:**
```python
np.savez(
    'crypto_5m_data.npz',
    train_price=train_price,
    train_tech=train_tech,
    train_date=train_date,
    test_price=test_price,
    test_tech=test_tech,
    test_date=test_date,
    crypto_pairs=CRYPTO_PAIRS,
    tech_indicators=TECH_INDICATORS
)
```

**수정된 코드:**
```python
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
    train_ohlcv_array=train_features['ohlcv_array'],
    train_ema_array=train_features['ema_array'],
    train_rsi_array=train_features['rsi_array'],
    train_stochastic_array=train_features['stochastic_array'],
    train_bb_percent_b_array=train_features['bb_percent_b_array'],
    train_price_change_array=train_features['price_change_array'],
    train_volume_change_array=train_features['volume_change_array'],
    train_obi_array=train_features['obi_array'],
    train_funding_rate_array=train_features['funding_rate_array'],
    test_ohlcv_array=test_features['ohlcv_array'],
    test_ema_array=test_features['ema_array'],
    test_rsi_array=test_features['rsi_array'],
    test_stochastic_array=test_features['stochastic_array'],
    test_bb_percent_b_array=test_features['bb_percent_b_array'],
    test_price_change_array=test_features['price_change_array'],
    test_volume_change_array=test_features['volume_change_array'],
    test_obi_array=test_features['obi_array'],
    test_funding_rate_array=test_features['funding_rate_array'],
    # 통계값들 (학습/테스트 공통)
    price_change_mean=train_features['price_change_mean'],
    price_change_std=train_features['price_change_std'],
    volume_change_mean=train_features['volume_change_mean'],
    volume_change_std=train_features['volume_change_std'],
)
```

---

## 2단계: 환경 생성 (Stock_NeurIPS2018_2_Train.ipynb)

### 기존 코드 위치
`Stock_NeurIPS2018_2_Train.ipynb`의 "Part 2. Build A Market Environment" 섹션

### 수정할 코드

**기존 코드:**
```python
# Part 1에서 저장한 데이터 로드
data = np.load('crypto_5m_data.npz', allow_pickle=True)

train_price = data['train_price']
train_tech = data['train_tech']
# ...

# 학습 환경 생성
train_config = {
    "price_array": train_price,
    "tech_array": train_tech,
}

env_train = CryptoEnv(
    config=train_config,
    lookback=1,
    initial_capital=1_000_000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
)
```

**수정된 코드:**
```python
# Part 1에서 저장한 데이터 로드
data = np.load('crypto_5m_data.npz', allow_pickle=True)

train_price = data['train_price']
train_tech = data['train_tech']
# ...

# 학습 환경 생성
train_config = {
    "price_array": train_price,
    "tech_array": train_tech,
    # 새로운 피처들 추가 (없으면 None)
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

env_train = CryptoEnv(
    config=train_config,
    lookback=1,
    initial_capital=1_000_000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
    window_size=84,  # 7시간 (84개 5분봉)
)
```

---

## 요약

1. **새로운 파일을 만들지 않습니다** - 기존 노트북 파일에 코드를 추가합니다.
2. **데이터 전처리 단계** (`Stock_NeurIPS2018_1_Data.ipynb`):
   - 새로운 피처 계산 함수 추가
   - 기존 데이터 처리 코드에 피처 계산 및 분리 코드 추가
   - 저장 코드에 새로운 피처들 추가
3. **환경 생성 단계** (`Stock_NeurIPS2018_2_Train.ipynb`):
   - 데이터 로드 시 새로운 피처들도 로드
   - `train_config` 딕셔너리에 새로운 피처들 추가

이렇게 하면 환경이 새로운 피처들을 사용할 수 있습니다!
