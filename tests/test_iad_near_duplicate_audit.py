from __future__ import annotations

import torch

from scripts.audit_iad_near_duplicates import nearest_train_matches


def test_nearest_train_matches_finds_raw_and_affine_matches() -> None:
    train = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[3.0, 1.0], [0.0, 2.0]]],
        ],
        dtype=torch.float64,
    )
    test = torch.stack(
        (
            train[1].clone(),
            2.0 * train[0] + 5.0,
        )
    )

    rmse, rmse_index, correlation, correlation_index = nearest_train_matches(
        train, test, block_size=1
    )

    assert rmse_index.tolist() == [1, 0]
    assert torch.isclose(rmse[0], torch.tensor(0.0, dtype=torch.float64))
    assert correlation_index.tolist() == [1, 0]
    assert torch.allclose(correlation, torch.ones(2, dtype=torch.float64))
