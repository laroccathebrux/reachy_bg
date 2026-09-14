from src.robot.sway import envelope, pose_at


def test_envelope_attacks_fast_and_releases_slowly():
    e = envelope(-20.0, 0.0)
    assert e == 0.5  # halfway to full loudness in one step
    e2 = envelope(-100.0, e)
    assert 0.4 < e2 < e  # slow release
    assert envelope(-100.0, 0.0) == 0.0
    assert envelope(0.0, 1.0) == 1.0


def test_pose_scales_with_amount_and_stays_small():
    assert pose_at(0.3, 0.0) == (0.0, 0.0, 0.0)
    pitch, yaw, roll = pose_at(0.3, 1.0)
    assert abs(pitch) <= 4.0 and abs(yaw) <= 6.0 and abs(roll) <= 2.5
    assert pose_at(0.3, 1.0) != pose_at(0.7, 1.0)
