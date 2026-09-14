"""Offline tests for hard-cut detection (synthetic frames)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from speaker_attribution.faces import is_hard_cut  # noqa: E402


def _feats(img):
    small = cv2.resize(img, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _scene(seed, base):
    rng = np.random.default_rng(seed)
    img = np.full((180, 320, 3), base, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 300), rng.integers(0, 160)
        col = tuple(int(c) for c in rng.integers(0, 120, 3))
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 20), col, -1)
    return img


def test_dark_to_dark_cut_detected():
    # two different dark scenes: hue correlation stays high, content changes
    a, b = _feats(_scene(1, (25, 20, 30))), _feats(_scene(2, (30, 25, 25)))
    assert is_hard_cut(a[0], b[0], a[1], b[1])


def test_same_scene_small_motion_not_a_cut():
    img = _scene(3, (25, 20, 30))
    moved = np.roll(img, 2, axis=1)
    a, b = _feats(img), _feats(moved)
    assert not is_hard_cut(a[0], b[0], a[1], b[1])


def test_first_frame_never_a_cut():
    a = _feats(_scene(4, (40, 40, 40)))
    assert not is_hard_cut(None, a[0], None, a[1])
