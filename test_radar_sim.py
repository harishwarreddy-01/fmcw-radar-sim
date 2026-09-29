import radar_sim as rs


def test_resolution():
    assert abs(rs.RANGE_RES - 0.0375) < 1e-6


def test_detects_moving_targets_and_removes_wall():
    targets = [(2.0, 0.8, 1.0), (3.5, -0.5, 0.6), (5.0, 0.0, 3.0)]
    dets, _ = rs.detect(targets)
    for r0, v0, _ in targets[:2]:
        assert any(abs(r - r0) < 0.1 and abs(v - v0) < 0.1 for r, v in dets), (r0, v0, dets)
    # static wall must be removed
    assert not any(abs(r - 5.0) < 0.1 and abs(v) < 0.05 for r, v in dets)
    assert len(dets) == 2
