from aft.contracts import DiagnosisReport, OutlierStats, State


def test_state_serialization_round_trip():
    report = DiagnosisReport(
        batch_id="batch-123",
        outlier_indices=[1, 3],
        outlier_stats=OutlierStats(median_grad_norm=0.5, max_grad_norm=12.3),
        suspected_samples=["s1", "s2"],
    )

    state = State(
        run_id="run-1",
        step=42,
        last_checkpoint_path="checkpoints/step-42",
        diagnosis_report=report,
        repair_attempts=2,
        blacklist=["bad-1"],
    )

    payload = state.to_dict()
    restored = State.from_dict(payload)

    assert restored.to_dict() == payload
