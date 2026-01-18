# Config Schema

This document describes the minimal configuration used by the system. The canonical schema is defined in `src/aft/config.py`.

## Sections

### training
- `steps_per_window`: number of steps per training window
- `batch_size`: batch size used by the data loader
- `lr`: learning rate
- `grad_clip`: gradient clipping threshold

### monitoring
- `spike_sigma`: loss spike detection threshold (mean + N*sigma)
- `grad_norm_threshold`: gradient explosion threshold
- `window_size`: rolling window size for stats

### forensics
- `outlier_ratio`: ratio vs median for outlier detection
- `zscore_threshold`: z-score threshold for outlier detection

### recovery
- `max_retries`: max repair attempts before escalation

### checkpoints
- `interval`: steps between checkpoints
- `keep_last_n`: retention count
- `async_save`: enable async Orbax saves
- `dir`: checkpoint directory

### curriculum
- `eval_interval`: steps between eval runs
- `domains`: domain labels used in eval
- `weights`: domain mixing weights

### notifications
- `webhook_url`: optional webhook endpoint
- `slack_channel`: optional Slack channel

### persistence
- `backend`: `sqlite` or `postgres`
- `database_url`: DB connection URL

### metrics
- `stdout`: emit metrics to stdout
- `jsonl_path`: JSONL output path for metrics
