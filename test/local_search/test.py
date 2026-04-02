from app.local_search import cost, get_labels, l2_distance, sample
from app.eval import clustering_accuracy
import numpy as np
import pytest


def test_l2_distance():
    x = np.array([[1.0, 2.0]])
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    result = l2_distance(x, centers)
    assert result.shape == (2,)
    assert np.allclose(result, np.array([[5.0, 25.0]]))

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    centers = np.array([[0.0, 0.0]])
    result = l2_distance(x, centers)
    assert result.shape == (2,)

    x = np.array([[1.0, 2.0]])
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    result = l2_distance(x, centers)
    assert result.shape == (2,)


def test_sample():
    sequence = np.arange(12).reshape(4, 3)
    size = 1
    result = sample(sequence, size)
    assert result.shape == (3,)
    size = 2
    result = sample(sequence, size)
    assert result.shape == (size, 3)


def test_cost():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    result = cost(x, centers)
    assert np.isclose(result, 10.0)


def test_get_labels():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    result = get_labels(x, centers)
    assert result.shape == (2,)
    assert np.allclose(result, np.array([0, 1]))


def test_clustering_accuracy():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    acc = clustering_accuracy(y_true, y_pred)
    assert acc == 1.0

    y_pred = [0, 0, 1, 1]
    acc = clustering_accuracy(y_true, y_pred)
    assert acc == 1.0

    y_pred = [0, 1, 0, 1]
    acc = clustering_accuracy(y_true, y_pred)
    assert acc == 0.5
