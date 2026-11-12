from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from iad_detection.cflow_matrix import expand_tasks, load_matrix_config
from iad_detection.methods.cflow_official import OfficialCFLOW, positional_encoding_2d


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/iad_cflow_official_matrix.json"


class CFLOWMatrixTests(unittest.TestCase):
    def test_locked_matrix_has_stable_sixty_cell_layout(self) -> None:
        tasks = expand_tasks(load_matrix_config(CONFIG))
        self.assertEqual(len(tasks), 60)
        self.assertEqual(
            (tasks[0].input_size, tasks[0].modulation, tasks[0].seed),
            (64, "16QAM", 2026),
        )
        self.assertEqual(
            (tasks[20].input_size, tasks[20].modulation, tasks[20].seed),
            (128, "16QAM", 2026),
        )
        self.assertEqual(
            (tasks[40].input_size, tasks[40].modulation, tasks[40].seed),
            (256, "16QAM", 2026),
        )
        self.assertEqual(
            (tasks[-1].input_size, tasks[-1].modulation, tasks[-1].seed),
            (256, "QPSK", 2030),
        )

    def test_duplicate_cells_are_rejected(self) -> None:
        config = copy.deepcopy(load_matrix_config(CONFIG))
        config["matrix"]["seeds"] = [2026, 2026]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            expand_tasks(config)

    def test_calibration_fixed_scores_order_anomalies(self) -> None:
        calibration = [torch.tensor([[[0.0, -1.0], [-1.0, -1.0]]])]
        references = OfficialCFLOW.calibration_references(calibration)
        normal = [torch.zeros((1, 2, 2))]
        anomalous = [torch.full((1, 2, 2), -4.0)]
        normal_score = OfficialCFLOW.image_scores(
            normal, references, input_size=4
        )[0]
        anomalous_score = OfficialCFLOW.image_scores(
            anomalous, references, input_size=4
        )[0]
        self.assertGreater(anomalous_score, normal_score)

    def test_positional_encoding_shape_and_finiteness(self) -> None:
        value = positional_encoding_2d(128, 8, 4, torch.device("cpu"))
        self.assertEqual(tuple(value.shape), (128, 8, 4))
        self.assertTrue(bool(torch.isfinite(value).all()))


if __name__ == "__main__":
    unittest.main()
