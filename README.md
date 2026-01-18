# auto-fine-tuning

Self-healing fine-tuning for large language models using JAX (fast per-sample diagnostics) and LangGraph (stateful orchestration). The system detects training anomalies, runs forensic analysis to isolate toxic samples, applies automatic repairs, and escalates to human review when needed.

## Why
Fine-tuning can fail due to loss spikes, toxic data, catastrophic forgetting, or infra faults. This project turns the training loop into a supervised, cyclic process that can recover without manual babysitting.

## Key Ideas
- Per-sample diagnostics with JAX `vmap(grad)` + `jit`
- Cyclic control flow with LangGraph (Train -> Monitor -> Forensic -> Planner)
- Orbax async checkpoints for fast rollback
- Human-in-the-loop interrupts for safe escalation

## Docs
- `docs/RESEARCH-1.md`: background and motivation
- `docs/SPEC.md`: functional spec
- `docs/PLAN.md`: implementation plan with dependencies

## Status
Design/spec phase. No runnable code yet.

## Intended CLI
- `aft run --config config.yaml`
- `aft resume --run_id <id>`
- `aft status --run_id <id>`
- `aft interrupt --run_id <id>`

## License
TBD
