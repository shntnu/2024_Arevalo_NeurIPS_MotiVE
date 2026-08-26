import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from infer_full_orf_predictions import annotation_flags, deterministic_topk


def test_deterministic_topk_sorts_ties_by_candidate_identifier():
    scores = torch.tensor(
        [
            [9.0, 1.0],
            [8.0, 5.0],
            [8.0, 5.0],
            [8.0, 4.0],
            [7.0, 3.0],
        ]
    )

    values, indices = deterministic_topk(scores, 3, dimension=0)

    np.testing.assert_array_equal(indices[0], [0, 1, 2])
    np.testing.assert_array_equal(indices[1], [1, 2, 3])
    np.testing.assert_array_equal(values[0], [9.0, 8.0, 8.0])
    np.testing.assert_array_equal(values[1], [5.0, 5.0, 4.0])

    row_values, row_indices = deterministic_topk(scores.T, 3, dimension=1)
    np.testing.assert_array_equal(row_indices, indices)
    np.testing.assert_array_equal(row_values, values)


def test_annotation_flags_uses_sorted_numeric_pair_keys():
    known = np.array([2, 8, 13, 21])
    pairs = np.array([0, 2, 8, 9, 21, 22])

    np.testing.assert_array_equal(
        annotation_flags(pairs, known), [False, True, True, False, True, False]
    )
