from robustness.adaptive_thresholds import SensitivityController, SensitivityConfig, SLOs


def test_spam_spike_raises_thresholds_and_restores():
    cfg = SensitivityConfig(
        baseline_theta_g=1.0,
        baseline_theta_u=1.0,
        window_size=20,
        raise_factor=1.2,
        max_multiplier_of_baseline=2.0,
        decay_alpha=0.9,
        log_path="data/test_adaptive_thresholds.log",
    )
    ctrl = SensitivityController(config=cfg, slos=SLOs())

    # Start steady with non-spam
    for _ in range(10):
        ctrl.record_event(is_spam=False, latency_ms=100.0)
    th0 = ctrl.thresholds()
    assert th0.theta_g == 1.0 and th0.theta_u == 1.0

    # Introduce a spike: 7 spam out of next 10 (total window spam_rate 0.35)
    for _ in range(7):
        ctrl.record_event(is_spam=True, latency_ms=120.0)
    for _ in range(3):
        ctrl.record_event(is_spam=False, latency_ms=120.0)

    th_spike = ctrl.thresholds()
    assert th_spike.theta_g > 1.0 and th_spike.theta_u > 1.0

    # After spam subsides, thresholds decay back towards baseline
    for _ in range(30):
        ctrl.record_event(is_spam=False, latency_ms=90.0)
    th_restored = ctrl.thresholds()
    assert th_restored.theta_g <= th_spike.theta_g
    assert th_restored.theta_u <= th_spike.theta_u
    assert abs(th_restored.theta_g - 1.0) < 0.2
    assert abs(th_restored.theta_u - 1.0) < 0.2


def test_latency_breach_triggers_back_pressure():
    cfg = SensitivityConfig(
        baseline_theta_g=0.5,
        baseline_theta_u=0.5,
        window_size=20,
        log_path="data/test_adaptive_thresholds.log",
    )
    ctrl = SensitivityController(config=cfg, slos=SLOs(p50_ms=1000, p95_ms=2000))

    # Fill window with high latency to breach p95
    for _ in range(20):
        ctrl.record_event(is_spam=False, latency_ms=2500.0)

    pol = ctrl.policy()
    assert pol.active is True
    assert pol.heavy_ops_enabled is False
    assert pol.sampler_size >= 2

