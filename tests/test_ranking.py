import numpy as np

from analysis.ranking import (baseline_reference_set, inverse_rank, jaccard_consecutive,
                              jaccard_to_reference, rank_active_neurons, region_presence,
                              top_k_mask)


def test_silent_neurons_are_unranked():
    attrs = np.array([[0.0, 0.0, 0.5, 0.0, -0.9, 0.0]])
    ranks = rank_active_neurons(attrs)
    np.testing.assert_array_equal(ranks, [[0, 0, 2, 0, 1, 0]])


def test_ranks_match_loop_reference_for_active_neurons():
    rng = np.random.default_rng(0)
    attrs = rng.normal(size=(500, 40)) * (rng.random((500, 40)) < 0.3)
    ranks = rank_active_neurons(attrs, chunk_size=64)
    for t in range(attrs.shape[0]):
        active = np.flatnonzero(attrs[t])
        expected = np.empty(len(active), dtype=int)
        expected[np.argsort(-np.abs(attrs[t, active]), kind='stable')] = np.arange(1, len(active) + 1)
        np.testing.assert_array_equal(ranks[t, active], expected)
        assert np.all(ranks[t, attrs[t] == 0] == 0)


def test_jaccard_ignores_silent_neurons():
    # one neuron active in consecutive bins; the old code padded both top-K sets with the same
    # silent neurons and reported J close to 1 even when the active neurons differ
    attrs = np.zeros((3, 50))
    attrs[0, 3] = 1.0
    attrs[1, 7] = 1.0
    jac = jaccard_consecutive(top_k_mask(rank_active_neurons(attrs), k=40))
    assert jac[0] == 0.0          # {3} vs {7}
    assert jac[1] == 0.0          # {7} vs {}
    assert np.isnan(jac[2])       # last bin has no successor


def test_jaccard_edge_cases():
    mask = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_allclose(jaccard_consecutive(mask), [0.5, 0.0, np.nan, np.nan])
    np.testing.assert_allclose(jaccard_to_reference(mask, [True, False, False]), [0.5, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(jaccard_to_reference(mask, [False, False, False]), [0.0, 0.0, np.nan, np.nan])


def test_inverse_rank_and_regions():
    ranks = np.array([[0, 2, 1]], dtype=np.uint16)
    np.testing.assert_allclose(inverse_rank(ranks), [[0, 0.5, 1]])
    presence, regions = region_presence(np.array([[True, False, False]]), np.array(['B', 'A', 'B']))
    assert list(regions) == ['A', 'B']
    np.testing.assert_array_equal(presence, [[False, True]])


def test_baseline_reference_set_excludes_silent_neurons():
    attrs = np.zeros((20, 6))
    attrs[0:5, 4] = 2.0       # neuron 4 dominates the baseline of event 0
    attrs[10:15, 1] = 1.0     # neuron 1 is the only one active in the baseline of event 1
    ref = baseline_reference_set(attrs, event_start_indices=[0, 10], baseline_bins=(0, 5), k=3)
    np.testing.assert_array_equal(np.flatnonzero(ref), [1, 4])
