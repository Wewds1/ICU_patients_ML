# Tests Notes

The `tests` directory contains lightweight checks that validate the most important repository behaviors.

## Coverage today

- `test_features.py` checks category normalization, missingness flags, and feature creation
- `test_api.py` checks the FastAPI health endpoint

## Recommended next additions

- API prediction contract test using a fixed fixture row
- artifact existence test after training
- regression test for threshold selection and model-card generation

