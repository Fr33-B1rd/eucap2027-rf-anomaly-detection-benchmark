from __future__ import annotations

import numpy as np

from scripts.run_iad_rf_energy_baseline import (
    fit_robust_profile,
    robust_energy_score,
)


def test_robust_profile_uses_median_mad_and_scale_floor() -> None:
    values = np.asarray(
        [
            [1.0, 4.0, 7.0],
            [2.0, 4.0, 8.0],
            [3.0, 4.0, 100.0],
        ]
    )
    center, scale, floored = fit_robust_profile(
        values, consistency_constant=1.0, scale_floor=0.25
    )

    np.testing.assert_allclose(center, [2.0, 4.0, 8.0])
    np.testing.assert_allclose(scale, [1.0, 0.25, 1.0])
    assert floored == 1


def test_robust_energy_score_is_mean_squared_z() -> None:
    values = np.asarray([[1.0, 4.0], [3.0, 8.0]])
    center = np.asarray([1.0, 2.0])
    scale = np.asarray([1.0, 2.0])

    actual = robust_energy_score(values, center=center, scale=scale)

    np.testing.assert_allclose(actual, [0.5, 6.5])
