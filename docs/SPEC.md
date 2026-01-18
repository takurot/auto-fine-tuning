# JAX x LangGraph Fine-Tuning Tool Spec

## 1. Overview
LLM fine-tuning operations are fragile due to loss spikes, toxic samples, catastrophic forgetting, and infra faults. This tool provides an autonomous, self-healing training loop by combining:
- JAX for fast, per-sample diagnostics (vmap + jit) and explicit state management.
- LangGraph for cyclic orchestration, persistent state, and human-in-the-loop controls.

The system shifts fine-tuning from passive scripts to an agentic supervisor that monitors, diagnoses, and repairs training in near real time.

## 2. Goals
- Detect and recover from loss spikes without manual babysitting.
- Identify and excise toxic samples using per-sample gradients.
- Guard against catastrophic forgetting via continuous evaluation and dynamic curriculum.
- Survive infra errors and avoid reboot loops via stateful orchestration.

## 3. Non-Goals
- Building a full-scale MLOps platform (deployment, feature stores, etc.).
- Training foundation models from scratch.
- Replacing existing training libraries; this tool orchestrates and augments them.

## 4. Target Users
- ML engineers running long fine-tuning jobs on GPU/TPU.
- Applied research teams building domain-specific LLMs.
- MLOps operators who need reliable recovery and auditability.

## 5. System Architecture
Two main processes:
1) Supervisor (LangGraph) on host CPU
2) Worker (JAX training kernels) on accelerators

### 5.1 High-Level Flow
TrainNode -> MonitorNode -> (Normal -> TrainNode)
                           (Anomaly -> ForensicNode -> PlannerNode)
                                                     (AutoFix -> TrainNode)
                                                     (Escalate -> HumanReviewNode)

### 5.2 Key Components
- JAX Training Loop: pmap-based distributed training.
- Forensic Tool: analyze_batch using vmap(grad) + jit.
- Orbax Checkpointing: async save / fast rollback.
- LangGraph State: persistent, minimal, string-based pointers to heavy data.
- HITL Interface: interrupts for manual approval or override.

## 6. Functional Requirements

### 6.1 Training
- Execute training in fixed step windows (default: 100 steps).
- Collect metrics: loss, grad_norm, NaN flags, throughput.
- Persist checkpoints periodically (default: every 100 steps) via Orbax.

### 6.2 Monitoring
- Detect loss spikes using rolling window statistics.
  - Default rule: current_loss > mean + 3*sigma over last N steps (N=1000).
- Detect gradient explosion (grad_norm > threshold).
- Detect NaN/Inf in loss or parameters.

### 6.3 Forensics (Per-sample Diagnosis)
- On anomaly, reload the problematic checkpoint.
- Run per-sample loss + gradient norm:
  - Function: analyze_batch(params, batch) -> {losses, grad_norms}
  - Implemented with jax.vmap(jax.grad(loss_fn)) and jax.jit.
- Identify outliers:
  - Default: grad_norm > median * 100 or z-score > 6.0.
- Emit Diagnosis Report into LangGraph state.

### 6.4 Planning & Repair
Planner chooses among:
- A) Surgical removal: blacklist offending sample IDs and retry.
- B) Stabilize: reduce LR (e.g., x0.5), increase grad clip, retry.
- C) Escalate: if retries exceed limit or diagnosis ambiguous.

### 6.5 Human-in-the-Loop
- Interrupt execution on severe or repeated failures.
- Notify via Slack/Webhook with Diagnosis Report.
- Accept commands:
  - resume
  - stop
  - set_lr <value>
  - skip_batch
  - blacklist_samples [ids]

### 6.6 Dynamic Curriculum
- Periodic multi-domain evaluation (default: every 1000 steps).
- If forgetting detected (domain score drops > threshold), adjust mix:
  - Increase proportion of degraded domain data.
  - Keep batch shapes stable to avoid recompilation.

## 7. Non-Functional Requirements
- Per-sample analysis must complete within seconds on GPU/TPU.
- Supervisor state must survive process restarts.
- Training recovery should not require manual checkpoint selection.
- Minimize overhead: no heavy tensors inside LangGraph state.

## 8. LangGraph State Schema
State must be serializable and persistent.

```
State {
  run_id: string
  step: int
  last_checkpoint_path: string
  metrics_window: {
    loss_history: float[]
    grad_norm_history: float[]
  }
  anomaly: {
    type: "loss_spike" | "grad_explosion" | "nan" | null
    step: int | null
  }
  diagnosis_report: {
    batch_id: string
    outlier_indices: int[]
    outlier_stats: {
      median_grad_norm: float
      max_grad_norm: float
    }
    suspected_samples: string[]
  } | null
  repair_attempts: int
  blacklist: string[]
  curriculum_mix: {
    domain_weights: map<string, float>
  }
  human_interrupt: {
    pending: bool
    reason: string | null
  }
}
```

## 9. Node Specifications

### 9.1 TrainNode
- Inputs: state, config
- Actions:
  - Load checkpoint
  - Train for K steps
  - Update metrics history
  - Save new checkpoint
- Outputs: updated state

### 9.2 MonitorNode
- Inputs: metrics history
- Actions:
  - Evaluate anomaly rules
  - Decide next node

### 9.3 ForensicNode
- Inputs: checkpoint, batch_id
- Actions:
  - Run analyze_batch
  - Compute outlier stats
  - Update diagnosis_report

### 9.4 PlannerNode
- Inputs: diagnosis_report, repair_attempts
- Actions:
  - Decide repair strategy
  - Mutate state: blacklist, LR, grad_clip

### 9.5 HumanReviewNode
- Inputs: diagnosis_report, state
- Actions:
  - Emit notification
  - Interrupt graph
  - Resume on external command

## 10. Interfaces

### 10.1 CLI
- `aft run --config config.yaml`
- `aft resume --run_id <id>`
- `aft status --run_id <id>`
- `aft interrupt --run_id <id>`

### 10.2 Webhook / Slack
- POST payload containing run_id, anomaly type, diagnosis summary, suggested actions.

### 10.3 Config (YAML)
Key settings:
- training: steps_per_window, batch_size, lr, grad_clip
- monitoring: spike_sigma, grad_norm_threshold
- forensics: outlier_ratio, zscore_threshold
- recovery: max_retries
- checkpoints: interval, keep_last_n, async=true
- curriculum: eval_interval, domains, weights
- notifications: webhook_url, slack_channel

## 11. Data Handling
- Dataset entries must include stable IDs for blacklist actions.
- Blacklist stored in persistent state and exported for offline cleanup.
- Batch reconstruction must exclude blacklisted IDs.

## 12. Reliability & Recovery
- Orbax async checkpointing with rolling retention.
- Supervisor restart resumes from persisted state.
- Avoid reboot loops by escalating after N retries.

## 13. Observability
- Metrics:
  - loss, grad_norm, NaN_count, throughput, eval_scores
- Logs:
  - anomaly detection events
  - repair decisions
  - human interventions
- Artifacts:
  - diagnosis reports
  - blacklist snapshots

## 14. Security & Safety
- All external notifications must be optional and configurable.
- Ensure no training data is leaked in notifications; include only IDs and stats.

## 15. Implementation Phases
1) JAX Forensic Kernel (analyze_batch)
2) LangGraph Supervisor Prototype (StateGraph + persistence)
3) Orbax integration (async checkpoint + rollback)
4) HITL UI/notifications (Slack or simple Web UI)

