"""Tests for the crowd density analytics feature."""

import numpy as np

from core.analytics import CrowdDensityAnalytics


class DummyTrack:
    """Simple stand-in for tracker outputs exposing a tlwh attribute."""

    def __init__(self, tlwh):
        self.tlwh = np.array(tlwh, dtype=np.float32)


def test_crowd_density_generates_overlay_and_zones():
    analytics = CrowdDensityAnalytics(max_zones=2, heatmap_resolution=(16, 16), decay_lambda=0.9)
    tracks = [
        DummyTrack([10, 10, 20, 40]),
        DummyTrack([40, 12, 25, 35]),
        DummyTrack([150, 160, 30, 60]),
    ]

    analytics.update(tracks, (200, 200))
    overlay = analytics.get_heatmap_overlay((200, 200))

    assert overlay.shape == (200, 200, 3)
    zones = analytics.get()['zones']
    assert 1 <= len(zones) <= 2
    total_people = sum(int(zone['count']) for zone in zones)
    assert total_people == len(tracks)

    before_decay = float(analytics.heatmap.sum())
    analytics.update([], (200, 200))
    after_decay = float(analytics.heatmap.sum())
    assert after_decay <= before_decay
