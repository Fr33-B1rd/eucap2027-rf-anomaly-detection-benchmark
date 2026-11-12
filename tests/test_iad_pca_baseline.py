from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from scripts.run_iad_pca_baseline import reconstruction_mse


def test_fast_reconstruction_mse_matches_explicit_inverse_transform() -> None:
    rng = np.random.default_rng(7)
    fit = rng.normal(size=(40, 12)).astype(np.float32)
    test = rng.normal(size=(9, 12)).astype(np.float32)
    model = PCA(n_components=5, svd_solver="full").fit(fit)

    expected = np.mean((test - model.inverse_transform(model.transform(test))) ** 2, axis=1)
    actual = reconstruction_mse(model, test, batch_size=4)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
