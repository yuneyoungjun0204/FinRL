# 모델 구조 정보

## 모델 구조 정의 위치

### 1. 기본 구조 (Stable Baselines3 라이브러리 내부)

현재 코드에서 사용하는 방식:
```python
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
```

**위치**: Stable Baselines3 라이브러리 내부
- 경로: `venv/Lib/site-packages/stable_baselines3/common/policies.py`
- 또는: `venv/Lib/site-packages/stable_baselines3/ppo/policies.py`

### 2. 기본 네트워크 구조 (PPO의 경우)

**기본 구조**:
```
입력 레이어 (관측값 차원) 
  ↓
히든 레이어 1: 64 뉴런 + ReLU 활성화
  ↓
히든 레이어 2: 64 뉴런 + ReLU 활성화
  ↓
출력 레이어 (액션 차원)
```

**구체적인 구조**:
- **Policy Network (액션 결정)**:
  - Input: 관측값 (observation_dim)
  - Hidden Layer 1: 64 뉴런 + ReLU
  - Hidden Layer 2: 64 뉴런 + ReLU
  - Output: 액션 (action_dim)

- **Value Network (가치 추정)**:
  - Input: 관측값 (observation_dim)
  - Hidden Layer 1: 64 뉴런 + ReLU
  - Hidden Layer 2: 64 뉴런 + ReLU
  - Output: 가치 (1개 스칼라)

**총 레이어 수**: 3개 (입력 + 히든 2개 + 출력)

---

## 알고리즘별 기본 구조

### PPO / A2C
- 히든 레이어: **2개**
- 각 레이어 뉴런 수: **64개**
- 활성화 함수: **ReLU** (기본값)

### SAC
- 히든 레이어: **2개**
- 각 레이어 뉴런 수: **256개**
- 활성화 함수: **ReLU**

### DDPG / TD3
- 히든 레이어: **2개**
- 각 레이어 뉴런 수: **[400, 300]** (첫 번째 400, 두 번째 300)
- 활성화 함수: **ReLU**

---

## 커스텀 네트워크 구조 정의 방법

### 방법 1: policy_kwargs 사용 (권장)

```python
from torch import nn

# 커스텀 네트워크 구조 정의
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],  # Policy와 Value 네트워크 구조
    activation_fn=nn.Tanh,  # 활성화 함수 변경
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    ent_coef=0.1
)
```

**설명**:
- `pi=[128, 128, 64]`: Policy 네트워크 - 128 → 128 → 64 뉴런 (3개 히든 레이어)
- `vf=[128, 128, 64]`: Value 네트워크 - 128 → 128 → 64 뉴런 (3개 히든 레이어)
- `activation_fn`: 활성화 함수 (ReLU, Tanh, ELU 등)

### 방법 2: 공유 레이어 사용

```python
policy_kwargs = dict(
    net_arch=[128, 128, dict(pi=[64], vf=[64])],  # 공유 레이어 2개 + 분리된 레이어
    activation_fn=nn.ReLU,
)
```

**설명**:
- 공유 레이어: 128 → 128 (Policy와 Value가 공유)
- 분리된 레이어: Policy는 64, Value는 64

---

## 현재 코드에서 확인하는 방법

### 방법 1: 모델 생성 후 구조 출력

```python
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)

# 모델 구조 확인
print(model.policy)
print(model.policy.mlp_extractor)
print(model.policy.action_net)
print(model.policy.value_net)
```

### 방법 2: TensorBoard 사용

```python
from stable_baselines3.common.callbacks import TensorBoardCallback

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
model.learn(
    total_timesteps=1000,
    callback=TensorBoardCallback(log_dir="./tensorboard_logs/")
)

# TensorBoard 실행: tensorboard --logdir=./tensorboard_logs/
# 그래프 탭에서 네트워크 구조 확인 가능
```

---

## 요약

1. **기본 구조 위치**: Stable Baselines3 라이브러리 내부 (`venv/Lib/site-packages/stable_baselines3/`)
2. **기본 구조 (PPO)**:
   - 히든 레이어: 2개
   - 각 레이어: 64 뉴런
   - 활성화 함수: ReLU
3. **커스터마이징**: `policy_kwargs` 파라미터로 `net_arch` 지정
4. **구조 확인**: 모델 생성 후 `print(model.policy)` 또는 TensorBoard 사용
