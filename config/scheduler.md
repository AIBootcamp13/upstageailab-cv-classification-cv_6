
### CosineAnnealingWarmRestarts

#### EfficientNetB3Model 일 경우 괜찮은 것 같음
```yaml
scheduler:
  name: cosine_warm_restarts
  params:
    T_0: 10
    T_mult: 1
    eta_min: 0.000001
```

#### EfficientNetB5Model 일 경우
```yaml
scheduler:
  name: cosine_warm_restart
  params:
    T_0: 20             # 첫 번째 사이클 길이 (20 epochs)
    T_mult: 2           # 이후 리스타트마다 길이 2배로 증가 → 20 → 40 → 80...
    eta_min: 0.000001   # 최소 learning rate (너무 낮게 가지 않게)
```


### CosineAnnealingLR

```yaml
scheduler:
  name: cosine
  params:
    T_max: 10
    eta_min: 0.000001
```