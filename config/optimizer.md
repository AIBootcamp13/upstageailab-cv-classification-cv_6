### AdamW

논문에서 L2 규제를 줬을 때, Adam의 경우 제대로 학습이 안된다는 결과가 있다고함.
```yaml
optimizer:
  name: AdamW
  params: {
    LEARNING_RATE: 0.0001,
    weight_decay: 0.01, # L2 규제를 어느정도로 줄 것인지( 일반적으로 0.1, 0.01 이런식으로 테스트해본다고 함!)
  }
```

### Adam

```yaml
optimizer:
  name: AdamW
  params: {
    LEARNING_RATE: 0.0001,
    weight_decay: 0,
  }
```