from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baloto_ml.data.encoding import (
    decode_multi_hot,
    decode_one_hot,
    encode_draws,
    multi_hot,
    one_hot,
)


def test_multi_hot_marks_exact_positions() -> None:
    mh = multi_hot([[1, 5, 20, 42, 43]])
    assert mh.shape == (1, 43)
    assert mh.dtype == np.uint8
    assert np.flatnonzero(mh[0]).tolist() == [0, 4, 19, 41, 42]
    assert mh.sum() == 5


def test_multi_hot_order_does_not_matter() -> None:
    np.testing.assert_array_equal(multi_hot([[43, 1, 20, 5, 42]]), multi_hot([[1, 5, 20, 42, 43]]))


def test_one_hot() -> None:
    oh = one_hot([1, 16, 8])
    assert oh.shape == (3, 16)
    assert oh.sum(axis=1).tolist() == [1, 1, 1]
    assert oh[1, 15] == 1 and oh[2, 7] == 1


@pytest.mark.parametrize(
    "bad", [[[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 44]], [[1, 1, 2, 3, 4]], [1, 2, 3, 4, 5]]
)
def test_multi_hot_rejects_invalid(bad) -> None:
    with pytest.raises(ValueError):
        multi_hot(bad)


def test_one_hot_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        one_hot([17])


def test_round_trip(history: pd.DataFrame) -> None:
    y_balls, y_sb = encode_draws(history)
    assert y_balls.shape == (len(history), 43)
    assert y_sb.shape == (len(history), 16)
    assert (y_balls.sum(axis=1) == 5).all()
    assert (y_sb.sum(axis=1) == 1).all()
    balls = history[["b1", "b2", "b3", "b4", "b5"]].to_numpy()
    np.testing.assert_array_equal(decode_multi_hot(y_balls), balls)
    np.testing.assert_array_equal(decode_one_hot(y_sb), history["superbalota"].to_numpy())
