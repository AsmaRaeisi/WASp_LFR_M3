import pytest


def test_sequence_perfect():
    from cvl.dataset import OnlineTrackingBenchmark
    otb = OnlineTrackingBenchmark("/courses/TSBB19/otb_mini")
    sequence = otb[0]
    track_output = [e['bounding_box'] for e in sequence]
    error = otb.calculate_per_frame_iou(0, track_output)
    for e in error:
        assert e == 1
