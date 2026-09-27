| # | Function or class | Location | Jev correct? | Basis | P(Q1) | P(Q2) | P(Q3) | P(Q4) |
|---|---|---|---|---|---|---|---|---|
| 1 | `_load_validator` | `pykissembed/baselines_engine.py:48` | **no** | conformant; near-threshold noise | 69.0 | — | 13.0 | — |
| 2 | `BaselineEnvelope` *(class)* | `pykissembed/baselines_engine.py:72` | **no** | conformant; near-threshold noise | 66.0 | — | 11.0 | — |
| 3 | `is_v1_envelope` | `pykissembed/baselines_engine.py:92` | **yes** | missing `Parameters` section | — | 66.0 | 13.0 | — |
| 4 | `load_envelope` | `pykissembed/baselines_engine.py:110` | **no** | conformant; near-threshold noise | 28.0 | — | 16.0 | 66.0 |
| 5 | `save_envelope` | `pykissembed/baselines_engine.py:165` | **no** | conformant; near-threshold noise | 66.0 | — | 11.0 | — |
| 6 | `locked_envelope` | `pykissembed/baselines_engine.py:207` | **yes** | missing `Returns` section | 66.0 | — | 10.0 | — |
| 7 | `ratchet` | `pykissembed/baselines_engine.py:238` | **no** | conformant; near-threshold noise | 53.0 | — | 16.0 | — |
| 8 | `read_int` | `pykissembed/baselines_engine.py:277` | **no** | conformant; near-threshold noise | — | — | 38.0 | — |
| 9 | `read_float` | `pykissembed/baselines_engine.py:299` | **no** | conformant; Q2 noise (no comments to judge) | 69.0 | 64.0 | 29.0 | — |
| 10 | `_numeric_entries` | `pykissembed/baselines_engine.py:322` | **yes** | missing `Returns` section | — | — | 15.0 | — |
| 11 | `read_int_map` | `pykissembed/baselines_engine.py:352` | **no** | conformant; near-threshold noise | 68.0 | — | 27.0 | — |
| 12 | `read_float_map` | `pykissembed/baselines_engine.py:370` | **no** | conformant; near-threshold noise | — | — | 22.0 | — |
| 13 | `_DefaultConfig` *(class)* | `pykissembed/checks/code_complexity.py:41` | **yes** | missing `Attributes` section | 65.0 | — | 5.0 | — |
| 14 | `_load_callable` | `pykissembed/checks/code_complexity.py:72` | **no** | conformant; near-threshold noise | — | — | 45.0 | — |
| 15 | `_extract_items_with_docstrings` | `pykissembed/checks/code_complexity.py:98` | **yes** | missing `Parameters` section | — | — | 9.0 | — |
| 16 | `_is_overload_stub` | `pykissembed/checks/code_complexity.py:136` | **yes** | missing `Parameters` section | — | 63.0 | 11.0 | — |
| 17 | `_decorator_tail` | `pykissembed/checks/code_complexity.py:147` | **yes** | missing `Parameters` section | 64.0 | — | 13.0 | — |
| 18 | `_get_line_count` | `pykissembed/checks/code_complexity.py:161` | **no** | conformant; Q2 noise (no comments to judge) | — | 64.0 | 46.0 | — |
| 19 | `_get_cc` | `pykissembed/checks/code_complexity.py:178` | **yes** | missing `Parameters` section | 69.0 | — | 16.0 | — |
| 20 | `_get_cog` | `pykissembed/checks/code_complexity.py:213` | **yes** | missing `Parameters` section | — | — | 16.0 | — |
| 21 | `_get_mi` | `pykissembed/checks/code_complexity.py:246` | **yes** | missing `Parameters` section | — | — | 13.0 | — |
| 22 | `_collect_metric_results` | `pykissembed/checks/code_complexity.py:273` | **no** | conformant; near-threshold noise | — | — | 31.0 | — |
| 23 | `_record_excess_metric_baselines` | `pykissembed/checks/code_complexity.py:317` | **no** | conformant; near-threshold noise | — | — | 26.0 | — |
| 24 | `_complexity_failure_message` | `pykissembed/checks/code_complexity.py:341` | **no** | conformant; Q2 noise (no comments to judge) | — | 61.0 | 48.0 | — |
| 25 | `_locked_envelope` | `pykissembed/checks/code_complexity.py:386` | **yes** | missing `Returns` section | 60.0 | — | 5.0 | — |
| 26 | `TestDocstringCoverage` *(class)* | `pykissembed/checks/code_complexity.py:409` | **no** | conformant; near-threshold noise | 36.0 | — | 4.0 | — |
| 27 | `TestLineCount` *(class)* | `pykissembed/checks/code_complexity.py:458` | **no** | conformant; near-threshold noise | 34.0 | — | 4.0 | — |
| 28 | `TestCyclomaticComplexity` *(class)* | `pykissembed/checks/code_complexity.py:494` | **no** | conformant; near-threshold noise | 30.0 | — | 4.0 | — |
| 29 | `TestWrapperProliferation` *(class)* | `pykissembed/checks/code_complexity.py:566` | **no** | conformant; Q2 noise (no comments to judge) | 55.0 | 67.0 | 5.0 | — |
| 30 | `TestMaintainabilityIndex` *(class)* | `pykissembed/checks/code_complexity.py:600` | **no** | conformant; near-threshold noise | 27.0 | — | 3.0 | — |
| 31 | `test_docstring_coverage` | `pykissembed/checks/code_complexity.py:414` | **yes** | missing `Parameters` section | 30.0 | — | 3.0 | — |
| 32 | `test_file_line_counts` | `pykissembed/checks/code_complexity.py:463` | **yes** | missing `Parameters` section | 30.0 | — | 4.0 | — |
| 33 | `test_cyclomatic_complexity` | `pykissembed/checks/code_complexity.py:499` | **yes** | missing `Parameters` section | 27.0 | — | 4.0 | — |
| 34 | `test_wrapper_proliferation` | `pykissembed/checks/code_complexity.py:571` | **yes** | missing `Parameters` section | 40.0 | — | 3.0 | — |
| 35 | `test_maintainability_index` | `pykissembed/checks/code_complexity.py:605` | **yes** | missing `Parameters` section | 26.0 | — | 3.0 | — |
| 36 | `_extract_int` | `pykissembed/checks/code_similarity.py:69` | **no** | conformant; Q2 noise (no comments to judge) | — | 65.0 | 49.0 | — |
| 37 | `_extract_str_list` | `pykissembed/checks/code_similarity.py:98` | **no** | conformant; Q2 noise (no comments to judge) | 62.0 | 51.0 | 51.0 | — |
| 38 | `_extract_float` | `pykissembed/checks/code_similarity.py:131` | **no** | conformant; Q2 noise (no comments to judge) | — | 58.0 | 41.0 | — |
| 39 | `_run_similarity_test` | `pykissembed/checks/code_similarity.py:160` | **no** | conformant; near-threshold noise | 35.0 | — | 19.0 | — |
| 40 | `test_providers_parallel` | `pykissembed/checks/code_similarity.py:263` | **no** | conformant; near-threshold noise | 34.0 | — | 22.0 | — |
| 41 | `test_combined_similarity` | `pykissembed/checks/code_similarity.py:332` | **no** | conformant; Q2 noise (no comments to judge) | 48.0 | 69.0 | 23.0 | — |
| 42 | `_run_one` | `pykissembed/checks/code_similarity.py:290` | **yes** | no docstring present (Q0) | — | — | — | — |
| 43 | `CommentStats` *(class)* | `pykissembed/checks/comment_density.py:30` | **yes** | missing `Attributes` section | — | 56.0 | 5.0 | — |
| 44 | `_get_int_attr` | `pykissembed/checks/comment_density.py:38` | **no** | conformant; near-threshold noise | — | — | 30.0 | — |
| 45 | `_comment_density_from_source` | `pykissembed/checks/comment_density.py:67` | **yes** | missing `Parameters` section | 58.0 | — | 27.0 | 69.0 |
| 46 | `_code_body_lines` | `pykissembed/checks/comment_density.py:91` | **yes** | missing `Parameters` section | 52.0 | — | 16.0 | — |
| 47 | `_all_functions_short` | `pykissembed/checks/comment_density.py:121` | **yes** | missing `Parameters` section | 56.0 | — | 8.0 | — |
| 48 | `_file_stats` | `pykissembed/checks/comment_density.py:152` | **no** | conformant; near-threshold noise | 67.0 | — | 51.0 | — |
| 49 | `_read_per_file_bounds` | `pykissembed/checks/comment_density.py:173` | **no** | conformant; near-threshold noise | 61.0 | — | 37.0 | — |
| 50 | `_iter_density_files` | `pykissembed/checks/comment_density.py:203` | **yes** | missing `Returns` section | — | — | 24.0 | — |
| 51 | `TestCommentDensity` *(class)* | `pykissembed/checks/comment_density.py:223` | **no** | conformant; near-threshold noise | 34.0 | — | 4.0 | — |
| 52 | `test_comment_density` | `pykissembed/checks/comment_density.py:228` | **yes** | missing `Parameters` section | 39.0 | — | 3.0 | — |
| 53 | `DocstringViolation` *(class)* | `pykissembed/checks/docstring_format.py:23` | **yes** | missing `Attributes` section | — | — | 4.0 | — |
| 54 | `_run_ruff_docstring_check` | `pykissembed/checks/docstring_format.py:37` | **no** | conformant; near-threshold noise | 52.0 | — | 40.0 | — |
| 55 | `_collect_docstring_violations` | `pykissembed/checks/docstring_format.py:118` | **no** | conformant; near-threshold noise | — | — | 63.0 | — |
| 56 | `_group_violations_by_file` | `pykissembed/checks/docstring_format.py:137` | **no** | conformant; near-threshold noise | — | — | 69.0 | — |
| 57 | `_classify_docstring_violations` | `pykissembed/checks/docstring_format.py:158` | **no** | conformant; near-threshold noise | — | — | 46.0 | — |
| 58 | `_count_violations_by_code` | `pykissembed/checks/docstring_format.py:201` | **no** | conformant; near-threshold noise | — | — | 64.0 | — |
| 59 | `_violation_headers` | `pykissembed/checks/docstring_format.py:223` | **no** | conformant; Q2 noise (no comments to judge) | 58.0 | 68.0 | 63.0 | — |
| 60 | `_docstring_failure_message` | `pykissembed/checks/docstring_format.py:244` | **no** | conformant; near-threshold noise | — | — | 65.0 | — |
| 61 | `TestDocstringFormat` *(class)* | `pykissembed/checks/docstring_format.py:283` | **no** | conformant; near-threshold noise | 35.0 | — | 9.0 | — |
| 62 | `__str__` | `pykissembed/checks/docstring_format.py:33` | **yes** | no docstring present (Q0) | — | — | — | — |
| 63 | `test_docstring_format` | `pykissembed/checks/docstring_format.py:288` | **yes** | missing `Parameters` section | 26.0 | — | 3.0 | — |
| 64 | `_DefaultThresholds` *(class)* | `pykissembed/checks/jev_docstring_audit.py:53` | **yes** | missing `Attributes` section | — | — | 6.0 | — |
| 65 | `SymbolState` *(class)* | `pykissembed/checks/jev_docstring_audit.py:152` | **yes** | missing `Attributes` section | 54.0 | 69.0 | 4.0 | — |
| 66 | `_is_overload_stub` | `pykissembed/checks/jev_docstring_audit.py:163` | **yes** | missing `Parameters` section | — | — | 12.0 | — |
| 67 | `_extract_symbol_states` | `pykissembed/checks/jev_docstring_audit.py:181` | **no** | conformant; near-threshold noise | 63.0 | — | 39.0 | — |
| 68 | `_load_api_key` | `pykissembed/checks/jev_docstring_audit.py:235` | **no** | conformant; near-threshold noise | 57.0 | — | 28.0 | — |
| 69 | `_requests_api` | `pykissembed/checks/jev_docstring_audit.py:262` | **no** | conformant; near-threshold noise | — | — | 43.0 | — |
| 70 | `_is_retryable` | `pykissembed/checks/jev_docstring_audit.py:295` | **no** | conformant; near-threshold noise | — | — | 61.0 | — |
| 71 | `_parse_noul_answers` | `pykissembed/checks/jev_docstring_audit.py:321` | **no** | conformant; near-threshold noise | 68.0 | — | 43.0 | — |
| 72 | `_query_jev` | `pykissembed/checks/jev_docstring_audit.py:356` | **no** | conformant; near-threshold noise | 51.0 | — | 43.0 | — |
| 73 | `_evaluate_symbol` | `pykissembed/checks/jev_docstring_audit.py:409` | **no** | conformant; near-threshold noise | — | — | 34.0 | 69.0 |
| 74 | `_format_pct_detail` | `pykissembed/checks/jev_docstring_audit.py:477` | **no** | conformant; Q2 noise (no comments to judge) | 55.0 | 67.0 | 45.0 | — |
| 75 | `_reason_message` | `pykissembed/checks/jev_docstring_audit.py:498` | **no** | conformant; near-threshold noise | 65.0 | — | 49.0 | — |
| 76 | `_format_symbol_failure` | `pykissembed/checks/jev_docstring_audit.py:550` | **no** | conformant; near-threshold noise | 48.0 | — | 38.0 | 65.0 |
| 77 | `_locked_envelope` | `pykissembed/checks/jev_docstring_audit.py:582` | **yes** | missing `Returns` section | 62.0 | — | 6.0 | — |
| 78 | `_read_thresholds` | `pykissembed/checks/jev_docstring_audit.py:605` | **no** | conformant; near-threshold noise | 66.0 | — | 61.0 | — |
| 79 | `TestJevDocstringAudit` *(class)* | `pykissembed/checks/jev_docstring_audit.py:637` | **no** | conformant; near-threshold noise | 34.0 | — | 5.0 | — |
| 80 | `test_jev_docstring_audit` | `pykissembed/checks/jev_docstring_audit.py:642` | **yes** | missing `Parameters` section | 26.0 | — | 4.0 | — |
| 81 | `_FileDiagnostics` *(class)* | `pykissembed/checks/lint_typecheck.py:32` | **yes** | missing `Attributes` section | — | 63.0 | 4.0 | — |
| 82 | `_Report` *(class)* | `pykissembed/checks/lint_typecheck.py:39` | **yes** | missing `Attributes` section | — | — | 6.0 | — |
| 83 | `_text` | `pykissembed/checks/lint_typecheck.py:46` | **no** | conformant; Q2 noise (no comments to judge) | — | 64.0 | 36.0 | — |
| 84 | `_number` | `pykissembed/checks/lint_typecheck.py:65` | **no** | conformant; Q2 noise (no comments to judge) | — | 65.0 | 34.0 | — |
| 85 | `_nested` | `pykissembed/checks/lint_typecheck.py:84` | **no** | conformant; near-threshold noise | — | — | 26.0 | — |
| 86 | `_resolve_tool` | `pykissembed/checks/lint_typecheck.py:103` | **no** | conformant; near-threshold noise | — | — | 47.0 | — |
| 87 | `run_ruff` | `pykissembed/checks/lint_typecheck.py:121` | **no** | conformant; near-threshold noise | 34.0 | — | 23.0 | 68.0 |
| 88 | `run_pyright` | `pykissembed/checks/lint_typecheck.py:173` | **no** | conformant; near-threshold noise | 49.0 | — | 36.0 | — |
| 89 | `build_report` | `pykissembed/checks/lint_typecheck.py:212` | **no** | conformant; near-threshold noise | 51.0 | — | 25.0 | — |
| 90 | `test_no_lint_or_type_errors` | `pykissembed/checks/lint_typecheck.py:309` | **yes** | missing `Parameters` section | 27.0 | — | 3.0 | — |
| 91 | `_Violation` *(class)* | `pykissembed/checks/no_suppressions.py:62` | **yes** | missing `Attributes` section | 67.0 | 60.0 | 5.0 | — |
| 92 | `_iter_project_py_files` | `pykissembed/checks/no_suppressions.py:72` | **yes** | missing `Returns` section | 66.0 | — | 20.0 | — |
| 93 | `_comment_violations` | `pykissembed/checks/no_suppressions.py:95` | **no** | conformant; near-threshold noise | 52.0 | — | 43.0 | 64.0 |
| 94 | `_from_import_cast_names` | `pykissembed/checks/no_suppressions.py:140` | **no** | conformant; near-threshold noise | — | — | 30.0 | — |
| 95 | `_cast_names` | `pykissembed/checks/no_suppressions.py:164` | **no** | conformant; near-threshold noise | — | — | 25.0 | — |
| 96 | `_cast_violations` | `pykissembed/checks/no_suppressions.py:190` | **no** | conformant; near-threshold noise | 65.0 | — | 20.0 | — |
| 97 | `_scan_file` | `pykissembed/checks/no_suppressions.py:243` | **no** | conformant; near-threshold noise | 57.0 | — | 31.0 | — |
| 98 | `_find_violations` | `pykissembed/checks/no_suppressions.py:272` | **no** | conformant; Q2 noise (no comments to judge) | — | 69.0 | 41.0 | — |
| 99 | `test_no_suppressions_or_casts` | `pykissembed/checks/no_suppressions.py:292` | **no** | conformant; near-threshold noise | 56.0 | — | 6.0 | — |
| 100 | `_main_callback` | `pykissembed/cli.py:47` | **yes** | missing `Parameters` section | 65.0 | 59.0 | 9.0 | — |
| 101 | `check` | `pykissembed/cli.py:75` | **yes** | missing `Parameters` section | 62.0 | — | 5.0 | — |
| 102 | `ratchet_cmd` | `pykissembed/cli.py:110` | **yes** | missing `Parameters` section | 23.0 | — | 5.0 | 67.0 |
| 103 | `_compute_current_for` | `pykissembed/cli.py:163` | **no** | conformant; near-threshold noise | — | — | 27.0 | — |
| 104 | `providers_list` | `pykissembed/cli.py:205` | **no** | conformant; near-threshold noise | 66.0 | — | 9.0 | — |
| 105 | `populate_embeddings` | `pykissembed/cli.py:247` | **no** | conformant; near-threshold noise | 40.0 | — | 19.0 | — |
| 106 | `type_review` | `pykissembed/cli.py:315` | **no** | conformant; near-threshold noise | 25.0 | — | 18.0 | 51.0 |
| 107 | `init` | `pykissembed/cli.py:359` | **yes** | missing `Parameters` section | 43.0 | — | 5.0 | — |
| 108 | `_auto_detect_paths` | `pykissembed/cli.py:432` | **no** | conformant; near-threshold noise | 47.0 | — | 19.0 | 68.0 |
| 109 | `_parse_pyproject` | `pykissembed/cli.py:461` | **yes** | missing `Parameters` section | 67.0 | — | 8.0 | — |
| 110 | `_setuptools_source_paths` | `pykissembed/cli.py:475` | **yes** | missing `Parameters` section | 51.0 | 64.0 | 14.0 | — |
| 111 | `_hatch_source_paths` | `pykissembed/cli.py:488` | **yes** | missing `Parameters` section | 59.0 | 69.0 | 15.0 | — |
| 112 | `_toml_value` | `pykissembed/cli.py:500` | **yes** | missing `Parameters` section | 59.0 | — | 9.0 | — |
| 113 | `_path_values` | `pykissembed/cli.py:516` | **yes** | missing `Parameters` section | — | 60.0 | 7.0 | — |
| 114 | `_package_roots` | `pykissembed/cli.py:531` | **yes** | missing `Parameters` section | 60.0 | 68.0 | 7.0 | — |
| 115 | `_default_source_paths` | `pykissembed/cli.py:546` | **yes** | missing `Parameters` section | — | — | 9.0 | — |
| 116 | `PyqtestConfig` *(class)* | `pykissembed/config.py:20` | **no** | conformant; near-threshold noise | 68.0 | — | 11.0 | — |
| 117 | `_read_toml` | `pykissembed/config.py:91` | **no** | conformant; Q2 noise (no comments to judge) | — | 56.0 | 24.0 | — |
| 118 | `_coerce_str_list` | `pykissembed/config.py:110` | **no** | conformant; Q2 noise (no comments to judge) | — | 68.0 | 20.0 | — |
| 119 | `_require_str_list` | `pykissembed/config.py:144` | **no** | conformant; Q2 noise (no comments to judge) | 60.0 | 68.0 | 26.0 | — |
| 120 | `_require_nonnegative_int` | `pykissembed/config.py:170` | **no** | conformant; near-threshold noise | — | — | 23.0 | — |
| 121 | `_require_bool` | `pykissembed/config.py:196` | **no** | conformant; Q2 noise (no comments to judge) | — | 63.0 | 29.0 | — |
| 122 | `load_config` | `pykissembed/config.py:222` | **no** | conformant; near-threshold noise | 30.0 | — | 15.0 | 63.0 |
| 123 | `_auto_detect` | `pykissembed/config.py:314` | **yes** | missing `Parameters` section | 11.0 | — | 14.0 | 35.0 |
| 124 | `get_config` | `pykissembed/config.py:348` | **no** | conformant; near-threshold noise | 64.0 | — | 22.0 | — |
| 125 | `reset_config_cache` | `pykissembed/config.py:370` | **no** | conformant; near-threshold noise | — | — | 8.0 | — |
| 126 | `baseline_path` | `pykissembed/config.py:67` | **yes** | missing `Returns` section | — | 62.0 | 6.0 | — |
| 127 | `cache_path` | `pykissembed/config.py:72` | **yes** | missing `Returns` section | 60.0 | — | 5.0 | — |
| 128 | `resolved_paths` | `pykissembed/config.py:80` | **no** | conformant; Q2 noise (no comments to judge) | — | 56.0 | 23.0 | — |
| 129 | `pykissembed_config` | `pykissembed/conftest.py:20` | **no** | conformant; near-threshold noise | — | — | 18.0 | — |
| 130 | `baseline_factory` | `pykissembed/conftest.py:33` | **yes** | missing `Parameters` section | 65.0 | — | 12.0 | — |
| 131 | `_make` | `pykissembed/conftest.py:43` | **yes** | no docstring present (Q0) | — | — | — | — |
| 132 | `should_skip` | `pykissembed/paths.py:51` | **no** | conformant; near-threshold noise | — | — | 29.0 | — |
| 133 | `iter_py_files` | `pykissembed/paths.py:74` | **yes** | missing `Returns` section | 56.0 | — | 13.0 | — |
| 134 | `resolve_paths` | `pykissembed/paths.py:106` | **no** | conformant; near-threshold noise | 43.0 | — | 27.0 | — |
| 135 | `root` | `pykissembed/paths.py:130` | **no** | conformant; Q2 noise (no comments to judge) | — | 65.0 | 29.0 | — |
| 136 | `include_notebooks` | `pykissembed/paths.py:141` | **no** | conformant; near-threshold noise | — | — | 26.0 | — |
| 137 | `warn_non_utf8` | `pykissembed/paths.py:155` | **no** | conformant; near-threshold noise | — | — | 23.0 | — |
| 138 | `_load_callable` | `pykissembed/plugin.py:55` | **yes** | missing `Parameters` section | 56.0 | 64.0 | 9.0 | — |
| 139 | `_is_str_object_dict` | `pykissembed/plugin.py:76` | **yes** | missing `Parameters` section | — | 52.0 | 11.0 | — |
| 140 | `_is_function_info` | `pykissembed/plugin.py:87` | **yes** | missing `Parameters` section | — | — | 18.0 | — |
| 141 | `pytest_addoption` | `pykissembed/plugin.py:105` | **yes** | missing `Parameters` section | — | — | 3.0 | — |
| 142 | `update_baselines` | `pykissembed/plugin.py:140` | **yes** | missing `Parameters` section | — | 59.0 | 23.0 | — |
| 143 | `cached_only` | `pykissembed/plugin.py:152` | **yes** | missing `Parameters` section | — | — | 14.0 | — |
| 144 | `pykissembed_paths` | `pykissembed/plugin.py:169` | **no** | conformant; near-threshold noise | 64.0 | — | 16.0 | — |
| 145 | `shared_baselines` | `pykissembed/plugin.py:187` | **no** | conformant; near-threshold noise | 42.0 | — | 33.0 | — |
| 146 | `shared_functions` | `pykissembed/plugin.py:214` | **yes** | missing `Parameters` section | 36.0 | — | 9.0 | — |
| 147 | `pca_cache` | `pykissembed/plugin.py:244` | **no** | conformant; Q2 noise (no comments to judge) | — | 69.0 | 30.0 | — |
| 148 | `pytest_configure` | `pykissembed/plugin.py:256` | **yes** | missing `Parameters` section | 35.0 | — | 9.0 | 67.0 |
| 149 | `_decide_injection` | `pykissembed/plugin.py:331` | **no** | conformant; near-threshold noise | 67.0 | — | 15.0 | — |
| 150 | `_has_marker_filter` | `pykissembed/plugin.py:414` | **no** | conformant; Q2 noise (no comments to judge) | — | 55.0 | 38.0 | — |
| 151 | `_has_keyword_filter` | `pykissembed/plugin.py:430` | **no** | conformant; Q2 noise (no comments to judge) | — | 53.0 | 38.0 | — |
| 152 | `_has_deselect_filter` | `pykissembed/plugin.py:446` | **no** | conformant; Q2 noise (no comments to judge) | — | 52.0 | 30.0 | — |
| 153 | `_first_node_id` | `pykissembed/plugin.py:462` | **no** | conformant; Q2 noise (no comments to judge) | — | 60.0 | 40.0 | — |
| 154 | `_smart_restricted_target` | `pykissembed/plugin.py:478` | **no** | conformant; near-threshold noise | 65.0 | — | 30.0 | — |
| 155 | `_check_candidate` | `pykissembed/plugin.py:501` | **no** | conformant; near-threshold noise | — | — | 31.0 | — |
| 156 | `_is_same_file` | `pykissembed/plugin.py:523` | **no** | conformant; Q2 noise (no comments to judge) | 61.0 | 61.0 | 31.0 | — |
| 157 | `_already_targets_check_file` | `pykissembed/plugin.py:545` | **no** | conformant; Q2 noise (no comments to judge) | — | 68.0 | 30.0 | — |
| 158 | `_is_installed_check_path` | `pykissembed/plugin.py:563` | **no** | conformant; Q2 noise (no comments to judge) | 44.0 | 68.0 | 26.0 | — |
| 159 | `_checks_dir` | `pykissembed/plugin.py:591` | **no** | conformant; near-threshold noise | — | — | 19.0 | — |
| 160 | `pytest_collect_file` | `pykissembed/plugin.py:616` | **yes** | missing `Parameters` section | 68.0 | — | 5.0 | — |
| 161 | `Provider` *(class)* | `pykissembed/providers/base.py:21` | **no** | conformant; near-threshold noise | 69.0 | — | 11.0 | — |
| 162 | `embed` | `pykissembed/providers/base.py:47` | **yes** | missing `Parameters` section | 69.0 | 64.0 | 18.0 | — |
| 163 | `is_configured` | `pykissembed/providers/base.py:58` | **yes** | missing `Returns` section | — | — | 6.0 | — |
| 164 | `ProviderRegistry` *(class)* | `pykissembed/providers/registry.py:19` | **yes** | missing `Attributes` section | 52.0 | — | 20.0 | — |
| 165 | `discover_builtin` | `pykissembed/providers/registry.py:133` | **no** | conformant; near-threshold noise | — | — | 7.0 | — |
| 166 | `discover_all` | `pykissembed/providers/registry.py:141` | **no** | conformant; near-threshold noise | — | — | 33.0 | — |
| 167 | `get` | `pykissembed/providers/registry.py:155` | **yes** | missing `Parameters` section | — | — | 11.0 | — |
| 168 | `cache_key` | `pykissembed/providers/registry.py:173` | **yes** | missing `Parameters` section | — | — | 7.0 | — |
| 169 | `__init__` | `pykissembed/providers/registry.py:24` | **no** | conformant; Q2 noise (no comments to judge) | — | 38.0 | 8.0 | — |
| 170 | `register` | `pykissembed/providers/registry.py:28` | **no** | conformant; Q2 noise (no comments to judge) | — | 63.0 | 28.0 | — |
| 171 | `clear` | `pykissembed/providers/registry.py:38` | **no** | conformant; Q2 noise (no comments to judge) | — | 64.0 | 9.0 | — |
| 172 | `discover` | `pykissembed/providers/registry.py:42` | **no** | conformant; near-threshold noise | 43.0 | — | 6.0 | 67.0 |
| 173 | `all` | `pykissembed/providers/registry.py:74` | **no** | conformant; Q2 noise (no comments to judge) | — | 61.0 | 31.0 | — |
| 174 | `get` | `pykissembed/providers/registry.py:84` | **yes** | missing `Parameters` section | — | 51.0 | 12.0 | — |
| 175 | `__contains__` | `pykissembed/providers/registry.py:95` | **yes** | missing `Parameters` section | — | 55.0 | 13.0 | — |
| 176 | `__len__` | `pykissembed/providers/registry.py:105` | **no** | conformant; Q2 noise (no comments to judge) | — | 46.0 | 43.0 | — |
| 177 | `__repr__` | `pykissembed/providers/registry.py:116` | **no** | conformant; Q2 noise (no comments to judge) | — | 60.0 | 41.0 | — |
| 178 | `get_function_text` | `pykissembed/similarity/ast_helpers.py:24` | **no** | conformant; near-threshold noise | — | — | 25.0 | — |
| 179 | `normalize_ast_tokens` | `pykissembed/similarity/ast_helpers.py:49` | **no** | conformant; near-threshold noise | — | — | 21.0 | — |
| 180 | `compute_content_hash` | `pykissembed/similarity/ast_helpers.py:66` | **no** | conformant; Q2 noise (no comments to judge) | — | 57.0 | 41.0 | — |
| 181 | `extract_text_for_embedding` | `pykissembed/similarity/ast_helpers.py:82` | **no** | conformant; near-threshold noise | 68.0 | 68.0 | 24.0 | — |
| 182 | `_decorator_lines` | `pykissembed/similarity/ast_helpers.py:122` | **yes** | missing `Parameters` section | 40.0 | — | 10.0 | 63.0 |
| 183 | `_signature_lines` | `pykissembed/similarity/ast_helpers.py:145` | **yes** | missing `Parameters` section | 51.0 | — | 8.0 | — |
| 184 | `_comment_lines` | `pykissembed/similarity/ast_helpers.py:163` | **yes** | missing `Parameters` section | 19.0 | — | 12.0 | 56.0 |
| 185 | `_fallback_comment_lines` | `pykissembed/similarity/ast_helpers.py:190` | **yes** | missing `Parameters` section | 46.0 | — | 12.0 | — |
| 186 | `_append_unique_comment_lines` | `pykissembed/similarity/ast_helpers.py:205` | **yes** | missing `Parameters` section | — | 55.0 | 4.0 | — |
| 187 | `_count_executable_lines` | `pykissembed/similarity/ast_helpers.py:214` | **no** | conformant; near-threshold noise | — | — | 43.0 | — |
| 188 | `_extract_function_from_node` | `pykissembed/similarity/ast_helpers.py:263` | **no** | conformant; near-threshold noise | 62.0 | — | 24.0 | — |
| 189 | `_extract_functions_from_source` | `pykissembed/similarity/ast_helpers.py:333` | **no** | conformant; Q2 noise (no comments to judge) | 65.0 | 68.0 | 20.0 | — |
| 190 | `extract_function_infos` | `pykissembed/similarity/ast_helpers.py:374` | **no** | conformant; near-threshold noise | 32.0 | — | 20.0 | 62.0 |
| 191 | `collapse_scan_directories` | `pykissembed/similarity/ast_helpers.py:425` | **yes** | missing `Parameters` section | 69.0 | — | 8.0 | — |
| 192 | `extract_function_infos_from_directories` | `pykissembed/similarity/ast_helpers.py:445` | **no** | conformant; near-threshold noise | 53.0 | — | 28.0 | — |
| 193 | `extract_all_function_infos` | `pykissembed/similarity/ast_helpers.py:501` | **no** | conformant; near-threshold noise | 49.0 | — | 23.0 | 63.0 |
| 194 | `extract_function_infos_from_file` | `pykissembed/similarity/ast_helpers.py:519` | **no** | conformant; near-threshold noise | — | — | 32.0 | — |
| 195 | `_load_cached_embeddings` | `pykissembed/similarity/checks.py:95` | **no** | conformant; near-threshold noise | — | — | 26.0 | — |
| 196 | `_combined_member_gaps` | `pykissembed/similarity/checks.py:131` | **no** | conformant; near-threshold noise | 54.0 | — | 23.0 | — |
| 197 | `_missing_embeddings_advice` | `pykissembed/similarity/checks.py:167` | **no** | conformant; near-threshold noise | 64.0 | — | 31.0 | — |
| 198 | `_skip_missing_embeddings` | `pykissembed/similarity/checks.py:208` | **yes** | missing `Parameters` section | 48.0 | — | 4.0 | — |
| 199 | `_format_pair_violation` | `pykissembed/similarity/checks.py:238` | **no** | conformant; Q2 noise (no comments to judge) | — | 51.0 | 42.0 | — |
| 200 | `_format_neighbor_violation` | `pykissembed/similarity/checks.py:260` | **no** | conformant; Q2 noise (no comments to judge) | — | 68.0 | 40.0 | — |
| 201 | `_report_violations` | `pykissembed/similarity/checks.py:283` | **no** | conformant; near-threshold noise | 69.0 | — | 12.0 | — |
| 202 | `run_provider_similarity_checks` | `pykissembed/similarity/checks.py:359` | **no** | conformant; near-threshold noise | 53.0 | — | 11.0 | — |
| 203 | `_jina_uncached` | `pykissembed/similarity/checks.py:475` | **no** | conformant; Q2 noise (no comments to judge) | — | 68.0 | 30.0 | — |
| 204 | `_find_matrix_violations` | `pykissembed/similarity/checks.py:508` | **no** | conformant; near-threshold noise | 56.0 | — | 33.0 | — |
| 205 | `run_jina_similarity_checks` | `pykissembed/similarity/checks.py:580` | **no** | conformant; near-threshold noise | 51.0 | — | 10.0 | — |
| 206 | `_extract_config` | `pykissembed/similarity/checks.py:689` | **no** | conformant; Q2 noise (no comments to judge) | 55.0 | 61.0 | 44.0 | — |
| 207 | `_extract_pca_variance` | `pykissembed/similarity/checks.py:714` | **no** | conformant; near-threshold noise | — | — | 35.0 | — |
| 208 | `_extract_refactor_index_threshold` | `pykissembed/similarity/checks.py:750` | **no** | conformant; Q2 noise (no comments to judge) | — | 62.0 | 47.0 | — |
| 209 | `_extract_refactor_index_top_n` | `pykissembed/similarity/checks.py:775` | **no** | conformant; Q2 noise (no comments to judge) | — | 65.0 | 55.0 | — |
| 210 | `_extract_embedding_cache` | `pykissembed/similarity/checks.py:800` | **yes** | missing `Parameters` section | — | — | 10.0 | — |
| 211 | `_extract_excluded_pairs` | `pykissembed/similarity/checks.py:825` | **no** | conformant; near-threshold noise | — | — | 47.0 | — |
| 212 | `AnalyzerError` *(class)* | `pykissembed/similarity/complexity.py:21` | **no** | conformant; near-threshold noise | — | — | 11.0 | — |
| 213 | `call_analyzer` | `pykissembed/similarity/complexity.py:25` | **yes** | missing `Parameters` section | 50.0 | — | 9.0 | — |
| 214 | `_extract_block_tuple` | `pykissembed/similarity/complexity.py:49` | **no** | conformant; near-threshold noise | — | — | 40.0 | — |
| 215 | `_cc_complexities_from_source` | `pykissembed/similarity/complexity.py:84` | **no** | conformant; near-threshold noise | 64.0 | — | 39.0 | — |
| 216 | `_get_complexities` | `pykissembed/similarity/complexity.py:116` | **no** | conformant; near-threshold noise | 42.0 | — | 41.0 | — |
| 217 | `_scan_complexity_directory` | `pykissembed/similarity/complexity.py:178` | **no** | conformant; near-threshold noise | 64.0 | — | 28.0 | — |
| 218 | `load_complexity_maps` | `pykissembed/similarity/complexity.py:220` | **yes** | missing `Parameters` section | 64.0 | — | 8.0 | — |
| 219 | `load_all_complexity_maps` | `pykissembed/similarity/complexity.py:237` | **no** | conformant; near-threshold noise | 61.0 | — | 11.0 | — |
| 220 | `baselines_dir` | `pykissembed/similarity/constants.py:45` | **no** | conformant; Q2 noise (no comments to judge) | 48.0 | 59.0 | 35.0 | — |
| 221 | `baselines_file` | `pykissembed/similarity/constants.py:56` | **no** | conformant; Q2 noise (no comments to judge) | — | 52.0 | 31.0 | — |
| 222 | `function_hashes_file` | `pykissembed/similarity/constants.py:67` | **no** | conformant; Q2 noise (no comments to judge) | — | 52.0 | 36.0 | — |
| 223 | `_embedding_cache_file` | `pykissembed/similarity/constants.py:78` | **no** | conformant; Q2 noise (no comments to judge) | — | 66.0 | 30.0 | — |
| 224 | `EmbeddingResponseError` *(class)* | `pykissembed/similarity/embeddings.py:63` | **no** | conformant; near-threshold noise | — | — | 10.0 | — |
| 225 | `_require_callable` | `pykissembed/similarity/embeddings.py:67` | **yes** | missing `Parameters` section | 64.0 | 67.0 | 9.0 | — |
| 226 | `_require_exception_type` | `pykissembed/similarity/embeddings.py:87` | **yes** | missing `Parameters` section | 56.0 | 63.0 | 10.0 | — |
| 227 | `_requests_api` | `pykissembed/similarity/embeddings.py:107` | **no** | conformant; near-threshold noise | — | — | 28.0 | — |
| 228 | `_response_json` | `pykissembed/similarity/embeddings.py:132` | **yes** | missing `Parameters` section | 60.0 | — | 18.0 | — |
| 229 | `_get_tiktoken_encoding` | `pykissembed/similarity/embeddings.py:147` | **no** | conformant; near-threshold noise | — | — | 23.0 | — |
| 230 | `_truncate_to_token_limit` | `pykissembed/similarity/embeddings.py:168` | **no** | conformant; near-threshold noise | 34.0 | — | 20.0 | — |
| 231 | `_load_api_key_from_env` | `pykissembed/similarity/embeddings.py:199` | **no** | conformant; near-threshold noise | — | — | 37.0 | — |
| 232 | `_get_embeddings_with_retry` | `pykissembed/similarity/embeddings.py:249` | **no** | conformant; near-threshold noise | 49.0 | 58.0 | 16.0 | — |
| 233 | `_require_api_key` | `pykissembed/similarity/embeddings.py:303` | **no** | conformant; Q2 noise (no comments to judge) | — | 66.0 | 59.0 | — |
| 234 | `_build_jina_caller` | `pykissembed/similarity/embeddings.py:328` | **yes** | missing `Parameters` section | 55.0 | — | 8.0 | — |
| 235 | `_parse_unindexed_response` | `pykissembed/similarity/embeddings.py:385` | **yes** | missing `Parameters` section | — | — | 11.0 | — |
| 236 | `_parse_voyage_embedding_item` | `pykissembed/similarity/embeddings.py:430` | **no** | conformant; near-threshold noise | 65.0 | — | 25.0 | — |
| 237 | `_parse_voyage_response` | `pykissembed/similarity/embeddings.py:486` | **no** | conformant; near-threshold noise | 51.0 | — | 26.0 | — |
| 238 | `_build_voyage_caller` | `pykissembed/similarity/embeddings.py:532` | **yes** | missing `Parameters` section | 62.0 | 69.0 | 11.0 | — |
| 239 | `_build_provider_caller` | `pykissembed/similarity/embeddings.py:594` | **no** | conformant; near-threshold noise | 66.0 | — | 33.0 | — |
| 240 | `get_embeddings_batch` | `pykissembed/similarity/embeddings.py:827` | **no** | conformant; near-threshold noise | 63.0 | — | 24.0 | — |
| 241 | `compute_cosine_similarity` | `pykissembed/similarity/embeddings.py:879` | **no** | conformant; near-threshold noise | — | — | 49.0 | — |
| 242 | `_l2_normalize` | `pykissembed/similarity/embeddings.py:908` | **yes** | missing `Parameters` section | — | 69.0 | 13.0 | — |
| 243 | `jina_combined_members` | `pykissembed/similarity/embeddings.py:923` | **yes** | missing `Parameters` section | — | — | 6.0 | — |
| 244 | `compute_combined_embedding` | `pykissembed/similarity/embeddings.py:943` | **no** | conformant; near-threshold noise | — | — | 25.0 | — |
| 245 | `_is_float_embedding` | `pykissembed/similarity/embeddings.py:1024` | **no** | conformant; Q2 noise (no comments to judge) | 62.0 | 49.0 | 21.0 | — |
| 246 | `_is_str_object_dict` | `pykissembed/similarity/embeddings.py:1045` | **no** | conformant; Q2 noise (no comments to judge) | — | 53.0 | 24.0 | — |
| 247 | `_is_embedding_cache` | `pykissembed/similarity/embeddings.py:1066` | **no** | conformant; Q2 noise (no comments to judge) | — | 58.0 | 25.0 | — |
| 248 | `get_cached_embedding` | `pykissembed/similarity/embeddings.py:1088` | **no** | conformant; near-threshold noise | — | — | 36.0 | — |
| 249 | `_jina_request` | `pykissembed/similarity/embeddings.py:349` | **yes** | missing `Parameters` section | 35.0 | — | 11.0 | — |
| 250 | `_voyage_request` | `pykissembed/similarity/embeddings.py:551` | **yes** | missing `Parameters` section | 44.0 | 61.0 | 20.0 | — |
| 251 | `_voyage_is_retryable` | `pykissembed/similarity/embeddings.py:571` | **yes** | missing `Parameters` section | 64.0 | — | 32.0 | — |
| 252 | `_gemini_request` | `pykissembed/similarity/embeddings.py:637` | **no** | conformant; near-threshold noise | 64.0 | — | 38.0 | — |
| 253 | `_gemini_is_retryable` | `pykissembed/similarity/embeddings.py:687` | **no** | conformant; Q2 noise (no comments to judge) | — | 59.0 | 32.0 | — |
| 254 | `_openai_request` | `pykissembed/similarity/embeddings.py:719` | **yes** | missing `Parameters` section | 43.0 | 64.0 | 9.0 | — |
| 255 | `_openrouter_request` | `pykissembed/similarity/embeddings.py:762` | **no** | conformant; Q2 noise (no comments to judge) | 56.0 | 65.0 | 36.0 | — |
| 256 | `is_excluded_pair` | `pykissembed/similarity/exclusions.py:23` | **no** | conformant; near-threshold noise | 44.0 | — | 26.0 | — |
| 257 | `_as_embeddings_cache` | `pykissembed/similarity/file_split.py:20` | **no** | conformant; near-threshold noise | 66.0 | — | 24.0 | — |
| 258 | `_as_config` | `pykissembed/similarity/file_split.py:58` | **no** | conformant; near-threshold noise | — | — | 35.0 | — |
| 259 | `generate_file_split_proposal` | `pykissembed/similarity/file_split.py:78` | **no** | conformant; near-threshold noise | 47.0 | — | 33.0 | — |
| 260 | `_infer_dim` | `pykissembed/similarity/jina_similarity.py:29` | **no** | conformant; near-threshold noise | — | — | 36.0 | — |
| 261 | `_stacked` | `pykissembed/similarity/jina_similarity.py:49` | **no** | conformant; near-threshold noise | — | — | 28.0 | — |
| 262 | `_row_normalize` | `pykissembed/similarity/jina_similarity.py:85` | **no** | conformant; near-threshold noise | — | — | 33.0 | — |
| 263 | `build_symmetrized_matrix` | `pykissembed/similarity/jina_similarity.py:104` | **no** | conformant; near-threshold noise | — | — | 38.0 | — |
| 264 | `_CupyArray` *(class)* | `pykissembed/similarity/pca.py:31` | **no** | conformant; near-threshold noise | 57.0 | — | 7.0 | — |
| 265 | `_CupyModule` *(class)* | `pykissembed/similarity/pca.py:44` | **yes** | missing `Attributes` section | 52.0 | 63.0 | 5.0 | — |
| 266 | `_PCAEstimator` *(class)* | `pykissembed/similarity/pca.py:59` | **no** | conformant; near-threshold noise | 69.0 | — | 4.0 | — |
| 267 | `_as_pca_estimator` | `pykissembed/similarity/pca.py:77` | **no** | conformant; near-threshold noise | — | — | 23.0 | — |
| 268 | `_explained_variance_ratio` | `pykissembed/similarity/pca.py:106` | **no** | conformant; near-threshold noise | — | — | 29.0 | — |
| 269 | `_load_cupy_module` | `pykissembed/similarity/pca.py:133` | **no** | conformant; Q2 noise (no comments to judge) | 45.0 | 69.0 | 24.0 | 67.0 |
| 270 | `_is_floating_array` | `pykissembed/similarity/pca.py:153` | **no** | conformant; near-threshold noise | — | — | 45.0 | — |
| 271 | `_to_numpy_float_array` | `pykissembed/similarity/pca.py:171` | **no** | conformant; near-threshold noise | — | — | 56.0 | — |
| 272 | `_to_numpy_from_cupy` | `pykissembed/similarity/pca.py:200` | **no** | conformant; near-threshold noise | — | — | 47.0 | — |
| 273 | `_load_sklearn_pca_class` | `pykissembed/similarity/pca.py:227` | **no** | conformant; near-threshold noise | — | — | 31.0 | — |
| 274 | `_load_sklearn_kmeans_class` | `pykissembed/similarity/pca.py:248` | **no** | conformant; near-threshold noise | — | — | 25.0 | — |
| 275 | `_get_pca_class` | `pykissembed/similarity/pca.py:269` | **no** | conformant; near-threshold noise | 67.0 | — | 20.0 | — |
| 276 | `fit_pca` | `pykissembed/similarity/pca.py:295` | **no** | conformant; near-threshold noise | 47.0 | — | 35.0 | — |
| 277 | `transform_embeddings_with_pca` | `pykissembed/similarity/pca.py:403` | **no** | conformant; near-threshold noise | 62.0 | 63.0 | 18.0 | — |
| 278 | `_KMeansModel` *(class)* | `pykissembed/similarity/pca.py:477` | **no** | conformant; Q2 noise (no comments to judge) | — | 64.0 | 4.0 | — |
| 279 | `_make_kmeans` | `pykissembed/similarity/pca.py:485` | **no** | conformant; near-threshold noise | 68.0 | — | 32.0 | — |
| 280 | `cluster_functions_kmeans_with_pca` | `pykissembed/similarity/pca.py:542` | **no** | conformant; near-threshold noise | — | — | 28.0 | — |
| 281 | `get` | `pykissembed/similarity/pca.py:34` | **yes** | missing `Returns` section | — | 57.0 | 7.0 | — |
| 282 | `__getitem__` | `pykissembed/similarity/pca.py:38` | **yes** | missing `Parameters, Returns` section | 42.0 | 51.0 | 4.0 | — |
| 283 | `asarray` | `pykissembed/similarity/pca.py:49` | **yes** | missing `Parameters, Returns` section | 25.0 | 69.0 | 4.0 | 56.0 |
| 284 | `cumsum` | `pykissembed/similarity/pca.py:53` | **yes** | missing `Parameters, Returns` section | 28.0 | 67.0 | 4.0 | — |
| 285 | `fit` | `pykissembed/similarity/pca.py:68` | **yes** | missing `Parameters, Returns` section | 42.0 | — | 4.0 | — |
| 286 | `transform` | `pykissembed/similarity/pca.py:72` | **yes** | missing `Parameters, Returns` section | 37.0 | — | 4.0 | — |
| 287 | `fit_predict` | `pykissembed/similarity/pca.py:480` | **yes** | missing `Parameters, Returns` section | 39.0 | — | 4.0 | 60.0 |
| 288 | `PopulationError` *(class)* | `pykissembed/similarity/populate_embeddings.py:64` | **no** | conformant; near-threshold noise | — | — | 7.0 | — |
| 289 | `_ProviderRequestError` *(class)* | `pykissembed/similarity/populate_embeddings.py:68` | **no** | conformant; near-threshold noise | — | — | 11.0 | — |
| 290 | `_emit` | `pykissembed/similarity/populate_embeddings.py:72` | **yes** | missing `Parameters` section | — | 34.0 | 5.0 | — |
| 291 | `_request_embeddings` | `pykissembed/similarity/populate_embeddings.py:77` | **yes** | missing `Parameters` section | 43.0 | — | 8.0 | — |
| 292 | `_FunctionHashEntry` *(class)* | `pykissembed/similarity/populate_embeddings.py:106` | **yes** | missing `Attributes` section | — | 46.0 | 6.0 | — |
| 293 | `_get_embedding_cache` | `pykissembed/similarity/populate_embeddings.py:113` | **yes** | missing `Parameters` section | 63.0 | — | 10.0 | — |
| 294 | `_get_function_hashes` | `pykissembed/similarity/populate_embeddings.py:141` | **no** | conformant; near-threshold noise | — | — | 31.0 | — |
| 295 | `_ProviderCfg` *(class)* | `pykissembed/similarity/populate_embeddings.py:175` | **yes** | missing `Attributes` section | — | 58.0 | 5.0 | — |
| 296 | `_JinaCfg` *(class)* | `pykissembed/similarity/populate_embeddings.py:187` | **no** | conformant; near-threshold noise | — | — | 14.0 | — |
| 297 | `_find_uncached` | `pykissembed/similarity/populate_embeddings.py:211` | **no** | conformant; near-threshold noise | — | — | 27.0 | — |
| 298 | `_populate_provider` | `pykissembed/similarity/populate_embeddings.py:243` | **no** | conformant; near-threshold noise | 41.0 | — | 23.0 | — |
| 299 | `_jina_texts` | `pykissembed/similarity/populate_embeddings.py:316` | **yes** | missing `Parameters` section | 68.0 | — | 8.0 | — |
| 300 | `_populate_jina` | `pykissembed/similarity/populate_embeddings.py:337` | **no** | conformant; near-threshold noise | 37.0 | — | 16.0 | — |
| 301 | `cli_provider_name` | `pykissembed/similarity/populate_embeddings.py:520` | **yes** | missing `Parameters` section | 47.0 | — | 7.0 | 64.0 |
| 302 | `_missing_for_cache` | `pykissembed/similarity/populate_embeddings.py:541` | **yes** | missing `Parameters` section | 56.0 | — | 9.0 | — |
| 303 | `_combined_member_gaps` | `pykissembed/similarity/populate_embeddings.py:562` | **yes** | missing `Parameters` section | 44.0 | — | 9.0 | — |
| 304 | `_populate_combined` | `pykissembed/similarity/populate_embeddings.py:583` | **yes** | missing `Parameters` section | 56.0 | 67.0 | 13.0 | — |
| 305 | `_populate_combined_scoped` | `pykissembed/similarity/populate_embeddings.py:597` | **yes** | missing `Parameters` section | 45.0 | — | 6.0 | — |
| 306 | `_update_function_hashes` | `pykissembed/similarity/populate_embeddings.py:647` | **no** | conformant; near-threshold noise | 69.0 | — | 20.0 | — |
| 307 | `_synchronize_scanned_function_hashes` | `pykissembed/similarity/populate_embeddings.py:668` | **yes** | missing `Parameters` section | 46.0 | — | 3.0 | — |
| 308 | `_function_key_is_in_scopes` | `pykissembed/similarity/populate_embeddings.py:689` | **yes** | missing `Parameters` section | 43.0 | 65.0 | 8.0 | — |
| 309 | `_scoped_text_hashes` | `pykissembed/similarity/populate_embeddings.py:706` | **yes** | missing `Parameters` section | 37.0 | 68.0 | 11.0 | 65.0 |
| 310 | `get_provider_populator` | `pykissembed/similarity/populate_embeddings.py:746` | **yes** | missing `Parameters` section | — | — | 20.0 | — |
| 311 | `_require_canonical_provider` | `pykissembed/similarity/populate_embeddings.py:787` | **yes** | missing `Parameters` section | 52.0 | — | 8.0 | — |
| 312 | `_provider_cache_keys` | `pykissembed/similarity/populate_embeddings.py:819` | **yes** | missing `Parameters` section | — | — | 18.0 | — |
| 313 | `_missing_for_provider` | `pykissembed/similarity/populate_embeddings.py:834` | **yes** | missing `Parameters` section | 69.0 | 68.0 | 9.0 | — |
| 314 | `_configured_credential` | `pykissembed/similarity/populate_embeddings.py:857` | **yes** | missing `Parameters` section | 47.0 | 66.0 | 12.0 | 53.0 |
| 315 | `_attempt_network_provider` | `pykissembed/similarity/populate_embeddings.py:875` | **yes** | missing `Parameters` section | 44.0 | — | 9.0 | — |
| 316 | `_inspect_caches` | `pykissembed/similarity/populate_embeddings.py:902` | **yes** | missing `Parameters` section | 55.0 | — | 4.0 | — |
| 317 | `_populate_all` | `pykissembed/similarity/populate_embeddings.py:920` | **no** | conformant; near-threshold noise | 43.0 | — | 39.0 | — |
| 318 | `_resolve_scan_directories` | `pykissembed/similarity/populate_embeddings.py:982` | **yes** | missing `Parameters` section | 56.0 | — | 9.0 | — |
| 319 | `populate_provider_embeddings` | `pykissembed/similarity/populate_embeddings.py:1007` | **no** | conformant; near-threshold noise | 31.0 | — | 16.0 | — |
| 320 | `populate_embeddings` | `pykissembed/similarity/populate_embeddings.py:1093` | **yes** | missing `Parameters` section | 52.0 | — | 3.0 | — |
| 321 | `main` | `pykissembed/similarity/populate_embeddings.py:1102` | **no** | conformant; near-threshold noise | 56.0 | — | 4.0 | — |
| 322 | `_RefactorConfig` *(class)* | `pykissembed/similarity/refactor_index.py:35` | **yes** | missing `Attributes` section | — | 49.0 | 5.0 | — |
| 323 | `_as_str_object_mapping` | `pykissembed/similarity/refactor_index.py:43` | **no** | conformant; near-threshold noise | — | — | 32.0 | — |
| 324 | `_parse_refactor_config` | `pykissembed/similarity/refactor_index.py:73` | **no** | conformant; Q2 noise (no comments to judge) | 52.0 | 61.0 | 40.0 | — |
| 325 | `_parse_cached_embeddings` | `pykissembed/similarity/refactor_index.py:121` | **no** | conformant; near-threshold noise | 61.0 | — | 40.0 | — |
| 326 | `compute_similarity_matrix` | `pykissembed/similarity/refactor_index.py:152` | **no** | conformant; near-threshold noise | 61.0 | — | 46.0 | — |
| 327 | `compute_max_similarities` | `pykissembed/similarity/refactor_index.py:203` | **no** | conformant; Q2 noise (no comments to judge) | 34.0 | 64.0 | 58.0 | 48.0 |
| 328 | `compute_similarity_indices` | `pykissembed/similarity/refactor_index.py:219` | **no** | conformant; near-threshold noise | — | — | 47.0 | — |
| 329 | `compute_refactor_indices` | `pykissembed/similarity/refactor_index.py:241` | **no** | conformant; near-threshold noise | — | — | 35.0 | — |
| 330 | `_max_similarities_excluding` | `pykissembed/similarity/refactor_index.py:266` | **no** | conformant; near-threshold noise | — | — | 36.0 | — |
| 331 | `get_refactor_priority_message` | `pykissembed/similarity/refactor_index.py:328` | **no** | conformant; near-threshold noise | 55.0 | — | 32.0 | — |
| 332 | `get_refactor_priority_message_for_complexity` | `pykissembed/similarity/refactor_index.py:446` | **no** | conformant; near-threshold noise | 56.0 | — | 19.0 | — |
| 333 | `_as_float_list` | `pykissembed/similarity/storage.py:48` | **no** | conformant; near-threshold noise | — | — | 27.0 | — |
| 334 | `_as_embedding_cache` | `pykissembed/similarity/storage.py:80` | **no** | conformant; near-threshold noise | 58.0 | — | 35.0 | — |
| 335 | `_as_str_object_dict` | `pykissembed/similarity/storage.py:116` | **no** | conformant; near-threshold noise | — | — | 32.0 | — |
| 336 | `_get_cache` | `pykissembed/similarity/storage.py:146` | **no** | conformant; Q2 noise (no comments to judge) | 63.0 | 63.0 | 30.0 | — |
| 337 | `_compress_embedding` | `pykissembed/similarity/storage.py:164` | **no** | conformant; near-threshold noise | — | — | 41.0 | — |
| 338 | `_decompress_embedding` | `pykissembed/similarity/storage.py:186` | **no** | conformant; Q2 noise (no comments to judge) | — | 58.0 | 49.0 | — |
| 339 | `_load_compressed_embeddings` | `pykissembed/similarity/storage.py:202` | **no** | conformant; Q2 noise (no comments to judge) | 59.0 | 67.0 | 23.0 | — |
| 340 | `_save_compressed_embeddings` | `pykissembed/similarity/storage.py:224` | **no** | conformant; near-threshold noise | 65.0 | — | 14.0 | — |
| 341 | `_atomic_json_write` | `pykissembed/similarity/storage.py:244` | **no** | conformant; near-threshold noise | 50.0 | — | 12.0 | — |
| 342 | `HashType` *(class)* | `pykissembed/similarity/storage.py:293` | **yes** | missing `Attributes` section | — | — | 9.0 | — |
| 343 | `ProviderEntry` *(class)* | `pykissembed/similarity/storage.py:301` | **no** | conformant; near-threshold noise | — | — | 13.0 | — |
| 344 | `hash_field` | `pykissembed/similarity/storage.py:341` | **yes** | missing `Returns` section | 58.0 | 66.0 | 6.0 | — |
| 345 | `threshold_pair_key` | `pykissembed/similarity/storage.py:346` | **yes** | missing `Returns` section | — | 63.0 | 6.0 | — |
| 346 | `threshold_neighbor_key` | `pykissembed/similarity/storage.py:351` | **yes** | missing `Returns` section | — | 60.0 | 6.0 | — |
| 347 | `pca_variance_key` | `pykissembed/similarity/storage.py:356` | **yes** | missing `Returns` section | — | 61.0 | 5.0 | — |
| 348 | `_build_combined_row` | `pykissembed/similarity/storage.py:361` | **yes** | missing `Parameters` section | 61.0 | — | 6.0 | — |
| 349 | `EmbeddingRegistry` *(class)* | `pykissembed/similarity/storage.py:435` | **no** | conformant; near-threshold noise | 41.0 | — | 31.0 | — |
| 350 | `__init__` | `pykissembed/similarity/storage.py:447` | **no** | conformant; Q2 noise (no comments to judge) | 61.0 | 66.0 | 30.0 | — |
| 351 | `providers` | `pykissembed/similarity/storage.py:465` | **yes** | missing `Returns` section | 62.0 | 58.0 | 5.0 | — |
| 352 | `base_providers` | `pykissembed/similarity/storage.py:470` | **yes** | missing `Returns` section | 56.0 | 69.0 | 5.0 | — |
| 353 | `standalone_providers` | `pykissembed/similarity/storage.py:477` | **yes** | missing `Returns` section | 63.0 | — | 5.0 | — |
| 354 | `combined` | `pykissembed/similarity/storage.py:482` | **yes** | missing `Returns` section | 64.0 | 53.0 | 5.0 | — |
| 355 | `combined_dependencies` | `pykissembed/similarity/storage.py:487` | **yes** | missing `Returns` section | 35.0 | — | 4.0 | 59.0 |
| 356 | `standalone_dependencies` | `pykissembed/similarity/storage.py:492` | **yes** | missing `Returns` section | 69.0 | — | 4.0 | — |
| 357 | `files` | `pykissembed/similarity/storage.py:497` | **yes** | missing `Returns` section | — | 63.0 | 4.0 | — |
| 358 | `by_cache_key` | `pykissembed/similarity/storage.py:501` | **no** | conformant; Q2 noise (no comments to judge) | — | 54.0 | 29.0 | — |
| 359 | `_provider_cache_sets` | `pykissembed/similarity/storage.py:518` | **yes** | missing `Parameters` section | — | — | 9.0 | — |
| 360 | `audit` | `pykissembed/similarity/storage.py:545` | **no** | conformant; Q2 noise (no comments to judge) | 63.0 | 67.0 | 37.0 | — |
| 361 | `remove_orphans` | `pykissembed/similarity/storage.py:572` | **no** | conformant; Q2 noise (no comments to judge) | 37.0 | 66.0 | 29.0 | 64.0 |
| 362 | `rebuild_combined` | `pykissembed/similarity/storage.py:595` | **no** | conformant; near-threshold noise | 33.0 | — | 19.0 | 66.0 |
| 363 | `get_valid_hashes` | `pykissembed/similarity/storage.py:793` | **no** | conformant; near-threshold noise | 55.0 | — | 38.0 | — |
| 364 | `load_minimal_baselines` | `pykissembed/similarity/storage.py:848` | **no** | conformant; near-threshold noise | 67.0 | — | 20.0 | — |
| 365 | `load_provider_embeddings` | `pykissembed/similarity/storage.py:878` | **no** | conformant; near-threshold noise | 47.0 | — | 23.0 | 66.0 |
| 366 | `load_baselines` | `pykissembed/similarity/storage.py:922` | **no** | conformant; near-threshold noise | 54.0 | 49.0 | 32.0 | — |
| 367 | `merge_embedding_caches` | `pykissembed/similarity/storage.py:950` | **no** | conformant; near-threshold noise | 54.0 | — | 14.0 | — |
| 368 | `save_baselines` | `pykissembed/similarity/storage.py:980` | **no** | conformant; near-threshold noise | 45.0 | — | 21.0 | — |
| 369 | `_save_baselines_unlocked` | `pykissembed/similarity/storage.py:996` | **yes** | missing `Parameters` section | 61.0 | — | 4.0 | — |
| 370 | `PCAModel` *(class)* | `pykissembed/similarity/types.py:14` | **no** | conformant; near-threshold noise | — | — | 5.0 | — |
| 371 | `FunctionInfo` *(class)* | `pykissembed/similarity/types.py:28` | **no** | conformant; near-threshold noise | 56.0 | — | 7.0 | — |
| 372 | `transform` | `pykissembed/similarity/types.py:22` | **yes** | missing `Parameters, Returns` section | 48.0 | 64.0 | 5.0 | — |
| 373 | `sync_vscode_settings` | `pykissembed/vscode_settings.py:30` | **no** | conformant; near-threshold noise | 63.0 | — | 33.0 | — |
| 374 | `_render_fresh_file` | `pykissembed/vscode_settings.py:94` | **no** | conformant; near-threshold noise | — | — | 21.0 | — |
| 375 | `_tolerant_parse` | `pykissembed/vscode_settings.py:109` | **yes** | missing `Parameters` section | — | — | 8.0 | — |
| 376 | `_patch_key` | `pykissembed/vscode_settings.py:134` | **yes** | missing `Parameters` section | 58.0 | — | 5.0 | — |
| 377 | `_insert_key` | `pykissembed/vscode_settings.py:155` | **yes** | missing `Parameters` section | 62.0 | — | 6.0 | — |
| 378 | `_detect_indent` | `pykissembed/vscode_settings.py:178` | **yes** | missing `Parameters` section | 67.0 | — | 12.0 | — |
| 379 | `WrapperCandidate` *(class)* | `pykissembed/wrapper_analysis.py:37` | **yes** | missing `Attributes` section | 56.0 | — | 4.0 | — |
| 380 | `parse_source_files` | `pykissembed/wrapper_analysis.py:46` | **no** | conformant; near-threshold noise | 35.0 | — | 31.0 | — |
| 381 | `find_wrapper_candidates` | `pykissembed/wrapper_analysis.py:82` | **no** | conformant; near-threshold noise | — | — | 26.0 | — |
| 382 | `_wrapper_candidate` | `pykissembed/wrapper_analysis.py:130` | **yes** | missing `Parameters` section | 57.0 | — | 7.0 | — |
| 383 | `_forwarding_call` | `pykissembed/wrapper_analysis.py:167` | **yes** | missing `Parameters` section | 57.0 | — | 11.0 | — |
| 384 | `_executable_body` | `pykissembed/wrapper_analysis.py:190` | **yes** | missing `Parameters` section | — | — | 13.0 | — |
| 385 | `_is_unchanged_forwarding` | `pykissembed/wrapper_analysis.py:201` | **yes** | missing `Parameters` section | 44.0 | — | 12.0 | — |
| 386 | `_receiver_name` | `pykissembed/wrapper_analysis.py:224` | **yes** | missing `Parameters` section | — | 64.0 | 9.0 | — |
| 387 | `_matches_positional_arguments` | `pykissembed/wrapper_analysis.py:237` | **yes** | missing `Parameters` section | 63.0 | — | 9.0 | — |
| 388 | `_matches_keyword_arguments` | `pykissembed/wrapper_analysis.py:267` | **yes** | missing `Parameters` section | 53.0 | — | 10.0 | — |
| 389 | `_is_parameter_name` | `pykissembed/wrapper_analysis.py:297` | **yes** | missing `Parameters` section | — | 58.0 | 9.0 | — |
| 390 | `_call_counts` | `pykissembed/wrapper_analysis.py:308` | **yes** | missing `Parameters` section | 52.0 | — | 11.0 | — |
| 391 | `_call_terminal_name` | `pykissembed/wrapper_analysis.py:328` | **yes** | missing `Parameters` section | — | 66.0 | 11.0 | — |
| 392 | `_parent_map` | `pykissembed/wrapper_analysis.py:343` | **yes** | missing `Parameters` section | 62.0 | — | 10.0 | — |
| 393 | `_wrapper_identifier` | `pykissembed/wrapper_analysis.py:360` | **yes** | missing `Parameters` section | 46.0 | — | 6.0 | — |
| 394 | `_qualified_name` | `pykissembed/wrapper_analysis.py:381` | **yes** | missing `Parameters` section | 55.0 | 60.0 | 8.0 | — |
| 395 | `_is_exempt_wrapper` | `pykissembed/wrapper_analysis.py:401` | **yes** | missing `Parameters` section | 64.0 | — | 8.0 | — |
| 396 | `_has_exempt_decorator` | `pykissembed/wrapper_analysis.py:423` | **yes** | missing `Parameters` section | 37.0 | 66.0 | 8.0 | — |
| 397 | `decorator_name` | `pykissembed/wrapper_analysis.py:449` | **yes** | missing `Parameters` section | — | — | 9.0 | — |
| 398 | `_matches_any` | `pykissembed/wrapper_analysis.py:469` | **yes** | missing `Parameters` section | — | 47.0 | 7.0 | — |
