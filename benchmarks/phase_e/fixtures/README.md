# Phase E prompt / schema regression fixtures

When `PROMPT_VERSION` or system templates change, re-run:

```bat
pytest tests/test_inference_parse_validate.py tests/test_inference_mock.py -q
python benchmarks\phase_e\run_inference_bench.py --mock
```

Live GPU (optional):

```bat
python -m frontend.cli infer server start
python benchmarks\phase_e\run_inference_bench.py --http
```

## Files

| File | Purpose |
|---|---|
| `valid_observe_plan.json` | Canonical good model output |
| `invalid_action.json` | Illegal action must be rejected |
| `invalid_element.json` | Unknown element_id must be rejected |
| `truncated_raw.txt` | Incomplete JSON must not be action-ready |

Fixtures are loaded by unit tests and the mock bench path.
