from benchmarks.analyze_prefetch_trace import allocate_prefetch_rows, replay


def test_allocate_prefetch_rows_respects_global_constraint():
    scores = [
        [0.9, 0.8, 0.7, 0.6, 0.1],
        [0.9, 0.8, 0.7, 0.4, 0.3],
    ]
    budgets = allocate_prefetch_rows(scores, total=8, minimum=3, maximum=5)

    assert budgets == [4, 4]
    assert sum(budgets) == 8


def test_replay_global_prefetch_can_reduce_repairs_without_changing_routing():
    demands = [
        [[0.40, 0.30, 0.20, 0.10, 0.00],
         [0.25, 0.24, 0.20, 0.16, 0.15]],
        [[0.40, 0.30, 0.20, 0.10, 0.00],
         [0.25, 0.24, 0.20, 0.16, 0.15]],
    ]
    experts = [
        [[0, 1, 2], [0, 1, 2, 3]],
        [[0, 1, 2], [0, 1, 2, 3, 4]],
    ]

    result = replay(
        demands, experts, budget=4, minimum=3, maximum=5, ema_decay=0.8,
    )

    assert result["uniform_misses"] == 1
    assert result["global_misses"] == 0
    assert result["repair_reduction"] == 1.0