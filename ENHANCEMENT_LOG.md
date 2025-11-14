# Gated Fusion Transformer Enhancement Log

## 날짜: 2025년 11월 14일

## ✅ 구현 완료: Multi-Modal Query Fusion

### 📌 주요 변경사항

#### 1. **새로운 Query Gate 추가** (`ResidualAttentionBlock_Gate.__init__`)

```python
# New: Query fusion gate for multi-modal query (video + text fusion)
self.query_gate = nn.Sequential(OrderedDict([
    ("qg_fc", nn.Linear(int(d_model * 3), int(d_model * 0.5), bias = False)),
    ("qg_gelu", QuickGELU()),
    ("qg_proj", nn.Linear(int(d_model * 0.5), 1, bias = False))
]))
```

**목적**: 비디오와 텍스트 임베딩을 동적으로 융합하기 위한 게이팅 함수

**입력**: 
- Video mean (x_mean)
- Audio mean (v_mean) 
- Text mean (t_mean)

**출력**: 0~1 사이의 가중치 값 (sigmoid 활성화)

---

#### 2. **Forward 함수 수정** (`ResidualAttentionBlock_Gate.forward`)

##### 기존 방식:
```python
# Cross attention에 원본 비디오 임베딩 직접 사용
x = x + self.cross_attention(self.ln_3(x), v, attn_mask/100) * attn_gate
```

##### 개선된 방식:
```python
# Step 1: Query gate 가중치 계산 (0~1 값)
query_gate_weight = self.query_gate(torch.cat((x_mean, v_mean, t_mean), dim=1)).sigmoid()

# Step 2: 비디오와 텍스트를 weighted sum으로 융합
fused_query = x * query_gate_weight + t * (1 - query_gate_weight)

# Step 3: 융합된 쿼리로 cross attention 수행
x = x + self.cross_attention(self.ln_3(fused_query), v, attn_mask/100) * attn_gate
```

---

### 🎯 기대 효과

1. **Adaptive Query Formation**: 텍스트 정보가 비디오 쿼리에 동적으로 반영됨
   - query_gate_weight ≈ 1.0: 비디오 중심 쿼리
   - query_gate_weight ≈ 0.0: 텍스트 중심 쿼리
   - query_gate_weight ≈ 0.5: 균형잡힌 융합

2. **Text-Guided Audio Fusion**: 텍스트가 오디오와 비디오 융합 과정에 직접 개입
   - 텍스트 쿼리가 오디오에서 관련 정보를 더 효과적으로 추출

3. **Enhanced Multi-Modal Interaction**: 3-way 상호작용 (Video ↔ Audio ↔ Text)
   - 기존: 비디오 → 오디오 (텍스트는 게이팅에만 사용)
   - 개선: (비디오+텍스트) → 오디오 (텍스트가 쿼리에도 사용)

---

### 🔧 기술적 세부사항

**Gate 계산 흐름:**
```
Input: x (video), v (audio), t (text)
       ↓
Pooling: x_mean, v_mean, t_mean
       ↓
Concat: [x_mean | v_mean | t_mean]  (shape: [batch, d_model*3])
       ↓
MLP: Linear(d_model*3 → d_model*0.5) → GELU → Linear(d_model*0.5 → 1)
       ↓
Sigmoid: query_gate_weight ∈ (0, 1)
       ↓
Fusion: fused_query = x ⊙ w + t ⊙ (1-w)
```

**파라미터 수 증가:**
- Query Gate MLP: `(d_model*3) * (d_model*0.5) + (d_model*0.5) * 1`
- 예시 (d_model=512): `1536 * 256 + 256 * 1 = 393,472` 파라미터

---

### 📝 코드 변경 위치

**파일**: `/Users/namgyu/Documents/3-2/AVIGATE_3/modules/module_cross.py`

1. **Import 추가** (Line 13)
   - `from typing import Tuple` 추가

2. **`__init__` 수정** (Line ~237)
   - `self.query_gate` 추가

3. **`forward` 수정** (Line ~263)
   - Multi-modal query fusion 로직 추가

---

### ⚠️ 주의사항

1. **기존 체크포인트와 호환성 없음**
   - 새로운 파라미터(`query_gate`)가 추가되어 기존 모델 로드 시 오류 발생
   - 해결책: 처음부터 재학습 또는 state_dict 수동 매핑 필요

2. **메모리 사용량**
   - Fused query를 저장하므로 약간의 추가 메모리 사용
   - 대부분의 경우 무시할 수준

3. **학습 안정성**
   - Sigmoid를 사용하므로 gradient vanishing 가능성 낮음
   - 초기화 중요: 기본 PyTorch 초기화 사용

---

## 🚀 다음 단계 (향후 구현 예정)

### Layer-Aware Gating
레이어 깊이에 따라 오디오 기여도를 동적으로 조정

```python
# 아이디어:
layer_ratio = current_layer / total_layers  # 0.0 ~ 1.0
layer_decay = 1.0 - layer_ratio
attn_gate = attn_gate * layer_decay  # 레이어가 깊어질수록 오디오 영향 감소
```

**장점**:
- 초반 레이어: Low-level 오디오 정보 적극 활용
- 후반 레이어: High-level semantic 정보에 집중

---

## 📊 실험 가이드

### 학습 시 확인할 지표

1. **Query Gate 값 분포**
   - 평균적으로 0.5 근처인지 (균형잡힌 융합)
   - 극단값(0 또는 1)에 치우쳤는지 확인

2. **기존 Gate들과의 상관관계**
   - `query_gate_weight` vs `attn_gate`
   - `query_gate_weight` vs `ff_gate`

3. **성능 변화**
   - R@1, R@5, R@10 메트릭
   - Text-to-Video vs Video-to-Text 성능 변화 비교

### 디버깅 팁

```python
# Forward 함수에 추가 가능한 디버그 코드:
if self.training and step % 100 == 0:
    print(f"Query gate: mean={query_gate_weight.mean():.3f}, "
          f"std={query_gate_weight.std():.3f}, "
          f"min={query_gate_weight.min():.3f}, "
          f"max={query_gate_weight.max():.3f}")
```

---

## ✍️ 작성자 노트

이번 수정으로 Gated Fusion Transformer의 표현력이 크게 향상될 것으로 기대됩니다. 
특히 텍스트가 단순히 게이팅 가중치를 결정하는 것을 넘어서, 
실제 쿼리 형성에도 직접 관여하게 되어 더 풍부한 multi-modal interaction이 가능해졌습니다.

**핵심 인사이트**: 
"좋은 질문(query)을 하려면 무엇을 찾고 있는지(text)를 알아야 한다"
