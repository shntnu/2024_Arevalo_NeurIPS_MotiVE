import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from infer_full_orf_predictions import known_flags, topk


def test_ranking_ties_and_annotation_flags():
    scores = torch.tensor([[9.0, 1.0], [8.0, 5.0], [8.0, 5.0], [8.0, 4.0], [7.0, 3.0]])
    values, indices = topk(scores, 3, dimension=0)
    np.testing.assert_array_equal(indices, [[0, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(values, [[9.0, 8.0, 8.0], [5.0, 5.0, 4.0]])
    row_values, row_indices = topk(scores.T, 3, dimension=1)
    np.testing.assert_array_equal(row_indices, indices)
    np.testing.assert_array_equal(row_values, values)
    np.testing.assert_array_equal(
        known_flags(np.array([0, 2, 8, 9, 21, 22]), np.array([2, 8, 13, 21])),
        [False, True, True, False, True, False],
    )
