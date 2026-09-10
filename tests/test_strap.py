"""Offline tests for the name-strap box detector (synthetic frames)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from speaker_attribution.chyron import _find_strap_box  # noqa: E402


def _busy(seed, h=360, w=640):
    # dense random texture, never near-white: stands in for foliage
    rng = np.random.default_rng(seed)
    return rng.integers(0, 160, size=(h, w, 3), dtype=np.uint8)


def _with_strap(img, text=True):
    cv2.rectangle(img, (64, 245), (165, 282), (235, 235, 235), -1)
    if text:
        cv2.putText(img, "Haley Robson", (68, 262),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        cv2.putText(img, "Accuser", (68, 278),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
    return img


def test_strap_found_on_busy_background():
    box = _find_strap_box(_with_strap(_busy(0)))
    assert box is not None
    x, y, w, h = box
    assert 55 <= x <= 75 and 235 <= y <= 255 and w >= 90


def test_no_strap_on_background_only():
    assert _find_strap_box(_busy(1)) is None


def test_blank_white_box_rejected():
    # a bright rectangle with no text inside is not a name strap
    assert _find_strap_box(_with_strap(_busy(2), text=False)) is None
