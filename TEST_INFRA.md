# E2E Test Infra: Quick Hopper Hybrid RAG

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA (Boundary Value Analysis) + Pairwise Combinatorial Testing + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Multiple PDF Upload Ingestion | ORIGINAL_REQUEST §F1 | 5 | 5 | ✓ |
| 2 | Document Source Attribution in Responses | ORIGINAL_REQUEST §F2 | 5 | 5 | ✓ |
| 3 | Feedback Mechanism (Thumbs Up/Down) | ORIGINAL_REQUEST §F3 | 5 | 5 | ✓ |
| 4 | Local Structured Log Storage | ORIGINAL_REQUEST §F4 | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**: `verify_enhancements.py` located at the project root, executed using standard Python `unittest` framework (e.g., `python verify_enhancements.py`).
- **Pass/Fail Semantics**: All assertions must pass (exit code 0 on success, non-zero on failure).
- **Test Case Format**: Each test case exercises functions in isolation (unit tests) and in coordination (integration tests), verifying state transitions, files, and outputs.
- **Directory Layout**:
  - `verify_enhancements.py`: Test runner and execution code.
  - `.agents/sub_orch_e2e_testing/test_design.md`: 4-Tier test case design.
  - `TEST_INFRA.md`: Overall test infrastructure plan and inventory.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Multi-Document Research Session | F1, F2, F3, F4 | High |
| 2 | Empty State to Ingestion Lifecycle | F1, F2, F3, F4 | Medium |
| 3 | Index Reset Preservation | F1, F4 | Medium |
| 4 | Multi-turn Cross-Document Reasoning | F1, F2, F4 | High |
| 5 | Robustness and Recovery under Partial Failures | F1, F2, F3, F4 | High |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥5 test cases per feature (Total: 20 test cases).
- **Tier 2 (Boundary & Corner Cases)**: ≥5 test cases per feature (Total: 20 test cases).
- **Tier 3 (Cross-Feature Combinations)**: Pairwise combinations covering interactions between F1, F2, F3, and F4 (4 test cases).
- **Tier 4 (Real-World Application Scenarios)**: 5 comprehensive end-to-end integration scenario test cases.
