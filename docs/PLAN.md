# Implementation Plan for JAX x LangGraph Fine-Tuning Tool

This plan breaks the SPEC into executable tasks with dependencies, deliverables, and acceptance checks. It assumes a Python codebase using JAX, LangGraph, Orbax, and a lightweight CLI.

## Progress Summary
- Task 0: completed (scaffolding decisions, config schema, sample config).
- Task 1: completed (data contracts, serialization helpers, docs/tests).
- Tasks 2-12: pending; next focus is Task 2 (Forensic kernel).

## Dependency Overview
0 -> 1 -> (2,3)
3 -> 4
(2,3,4) -> 5 -> 6 -> 7
(3,4) -> 8
(0,1,5) -> 9
(3,4,5,6,7,8) -> 10
(3,5,6,7) -> 11
(0-11) -> 12

## 0. Baseline Decisions and Scaffolding [DONE]
Goal: establish project conventions before core implementation.
Depends on: none

- Decide package layout (e.g., `src/aft/`) and module boundaries.
- Decide config format and schema validation library (e.g., pydantic/omegaconf).
- Decide storage for LangGraph persistence (SQLite for local, Postgres for prod).
- Decide metrics sink (stdout + JSONL + optional Prometheus).
- Produce a minimal `config.yaml` example aligned with SPEC.

Acceptance:
- Skeleton directories exist.
- Config schema documented and validated by a unit test.

## 1. Data Contracts [DONE]
Goal: define stable interfaces used across components.
Depends on: 0

Tasks:
- Define `State` dataclass (matches SPEC schema).
- Define `DiagnosisReport` dataclass.
- Define `Batch` protocol including `sample_id` array.
- Define `MetricsWindow` structure.
- Document serialization strategy (JSON-friendly, no tensors).

Acceptance:
- State serialization round-trip test passes.
- No heavy tensors in state.

## 2. JAX Forensic Kernel (analyze_batch) [TODO]
Goal: implement fast per-sample diagnostics.
Depends on: 1

Tasks:
- Implement `loss_fn(params, batch)` for a minimal model stub.
- Implement `analyze_batch(params, batch)` returning per-sample loss and grad norms.
- Add `jax.vmap(jax.grad(loss_fn))` and `jax.jit`.
- Add outlier detection helper (median*ratio, z-score).
- Provide a micro-benchmark script to measure runtime on GPU/TPU.

Acceptance:
- `analyze_batch` runs on a synthetic batch and returns correct shapes.
- Runtime within seconds on GPU for target batch size.

## 3. Training Loop + Orbax Checkpointing [TODO]
Goal: enable checkpointed training windows with rollback.
Depends on: 0, 1

Tasks:
- Implement training step (pmap-ready).
- Implement `train_window(state, K)` that runs K steps and collects metrics.
- Integrate Orbax CheckpointManager with async save.
- Implement restore to any prior step.

Acceptance:
- Training window produces checkpoints and can restore deterministically.
- Async checkpointing does not block training.

## 4. Monitoring and Anomaly Detection [TODO]
Goal: detect loss spikes, grad explosions, NaNs.
Depends on: 1, 3

Tasks:
- Implement rolling window stats for loss.
- Implement anomaly rules: loss spike, grad explosion, NaN/Inf.
- Add configurable thresholds from config.
- Emit anomaly events into state.

Acceptance:
- Synthetic tests trigger each anomaly type.
- MonitorNode selects correct branch for normal/anomalous states.

## 5. LangGraph Supervisor (StateGraph) [TODO]
Goal: orchestrate Train -> Monitor -> Forensic -> Planner -> (Train/HITL).
Depends on: 1, 2, 3, 4

Tasks:
- Build StateGraph with nodes: TrainNode, MonitorNode, ForensicNode, PlannerNode, HumanReviewNode.
- Wire conditional edges and retries.
- Add persistent state backend (SQLite default).
- Ensure state is loaded/resumed correctly on restart.

Acceptance:
- Graph runs end-to-end in a local demo.
- Restart resumes from saved state without data loss.

## 6. Planner and Repair Actions [TODO]
Goal: implement decision logic and corrective actions.
Depends on: 1, 2, 4, 5

Tasks:
- Implement repair strategy selection based on diagnosis and retry count.
- Implement blacklist update and batch reconstruction hook.
- Implement LR/grad-clip adjustment hooks.
- Add retry policy with max attempts.

Acceptance:
- Unit tests cover A/B/C planner branches.
- Blacklisted sample IDs are never reloaded into batches.

## 7. Human-in-the-Loop (HITL) [TODO]
Goal: pause and wait for operator input on severe issues.
Depends on: 5, 6

Tasks:
- Add interrupt handling in LangGraph.
- Implement notification stub (Webhook and Slack).
- Implement command handler: resume, stop, set_lr, skip_batch, blacklist_samples.

Acceptance:
- Manual command can resume a paused graph.
- Notifications omit raw data (IDs only).

## 8. Dynamic Curriculum [TODO]
Goal: prevent catastrophic forgetting during training.
Depends on: 1, 3, 4

Tasks:
- Implement evaluation hooks for multiple domains.
- Detect score drops and update `curriculum_mix`.
- Modify data loader to respect updated mix.

Acceptance:
- Simulated eval drop triggers mix change.
- Batch shapes remain stable across mix changes.

## 9. CLI and Config [TODO]
Goal: provide operational control.
Depends on: 0, 1, 5

Tasks:
- Implement CLI commands: `run`, `resume`, `status`, `interrupt`.
- Validate config on startup.
- Provide example config and README usage.

Acceptance:
- CLI can start and resume a run.
- Config errors are reported clearly.

## 10. Observability and Artifacts [TODO]
Goal: make runs auditable.
Depends on: 3, 4, 5, 6, 7, 8

Tasks:
- Log events (anomaly, repair, HITL) to JSONL.
- Export metrics (loss, grad_norm, eval) per step window.
- Save diagnosis reports and blacklist snapshots.

Acceptance:
- Artifacts are written with run_id scoping.
- Metrics and logs can be parsed post-run.

## 11. Reliability and Recovery Tests [TODO]
Goal: ensure system survives faults.
Depends on: 3, 5, 6, 7

Tasks:
- Simulate worker crash and supervisor restart.
- Simulate repeated anomaly causing retry loop and verify escalation.
- Verify checkpoint retention (keep_last_n).

Acceptance:
- Restart resumes without manual intervention.
- Escalation triggers after max retries.

## 12. Documentation [TODO]
Goal: make the system usable by others.
Depends on: 0-11

Tasks:
- Update `docs/SPEC.md` if implementation deviates.
- Write `docs/ARCHITECTURE.md` with diagrams.
- Add `docs/CONFIG.md` describing all fields.

Acceptance:
- Documentation matches actual code paths and config schema.

## Suggested Milestones
1) M1: Forensic kernel + minimal training window
2) M2: LangGraph supervisor with persistence
3) M3: Orbax rollback and anomaly detection
4) M4: Planner repairs + HITL
5) M5: Curriculum + CLI + docs
