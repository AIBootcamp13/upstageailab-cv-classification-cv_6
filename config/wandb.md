### sweep 명령어
```bash
wandb sweep ./config/sweep-config.yaml
```

### 백그라운드 실행
```bash
nohup ./run_agent.sh > wandb_agent.log 2>&1 &
```