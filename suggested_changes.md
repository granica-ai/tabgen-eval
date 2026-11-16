# TabGenEval — historical implementation review

This archived review describes an earlier implementation. Its API examples and
recommendations have been superseded; use the current [evaluation reference](docs/source/reference/evaluation.md)
and [architecture guide](docs/architecture.md) for supported behavior. Original
identifiers below are retained as historical context.

Running log of suggested changes from the unit-by-unit review, with reasoning.
Status legend: **NOTED** (agreed, not yet done) · **OPTIONAL** (low value, author's call) · **DONE** · **REJECTED**.

---

## Unit A — schema + config

Reviewed: `schema/{schema,infer,validate}.py`, `config.py`. No statistical-correctness
bugs found. One false positive (partial-schema column drop) was withdrawn — it's intended
design, covered by `test_unsupported_column_outside_schema_is_ignored`.

### A-1 (MEDIUM, NOTED) — `EvalConfig` doesn't range-check several fields the docs say it does
- **Where:** `config.py` `EvalConfig.__post_init__`
- **What:** `n_bootstrap` (unbounded, e.g. `10**9` accepted), `min_bootstrap_n` (≤0 accepted),
  `size_disparity_ratio` (≤1 accepted — would flag *every* comparison since `max/min ≥ 1`)
  are not validated.
- **Why it matters:** `docs/failure-modes.md:45` explicitly lists *"`EvalConfig` validates
  ranges"* as the mitigation for "Memory/time DoS via config". The code doesn't honor that
  claim. (`max_eval_n` still bounds per-metric cost, so impact is limited in practice.)
- **Fix:** add to `__post_init__` — `min_bootstrap_n >= 1`, `size_disparity_ratio >= 1`, and a
  sanity cap/warn on `n_bootstrap`. **Or** soften the doc claim to match the code.
  Leaning: add the checks (3 cheap lines, doc already promises them).

### A-2 (LOW, OPTIONAL) — nested-object validation guard samples only `head(50)`
- **Where:** `schema/validate.py` `_unsupported_kind` (~:111)
- **What:** a column that is scalar for its first 50 non-null rows but holds a
  `list`/`dict`/`ndarray` later passes validation.
- **Why it matters:** intent (`test_nested_object_column_raises`) is to *raise* on nested
  objects. Confirmed effect is **not** a crash — the column infers categorical and the
  fidelity/diversity/conditional metrics return **NaN** (degenerate, not wrong numbers). Rare
  in practice. Test only covers the all-rows-nested case.
- **Fix (if desired):** scan the full non-null column (one `apply(isinstance,...)` pass) instead
  of `head(50)`.

### A-3 (NIT, OPTIONAL) — `clean` / `categorical_nan_policy` literals not validated at the config boundary
- **Where:** `config.py`
- **What:** `EvalConfig(clean="bogus")` constructs fine and only raises later inside `cleaning`
  with a plain `ValueError`, not an actionable error at the boundary. (CLI is safe via `choices=`.)
- **Fix (if desired):** membership-check the `Literal` fields in `__post_init__`.

### A-4 (NIT, OPTIONAL) — 0-column DataFrame (rows > 0, no columns) passes `validate`
- **Where:** `schema/validate.py` empty check (~:47)
- **What:** check is `shape[0] == 0` only; a 0-column frame is accepted. Pathological input;
  doesn't crash. Consider raising when `shape[1] == 0` or when the effective schema is empty.

---

## Unit B — statistical primitives

Reviewed: `metrics/primitives.py` (KS-complement, TV-complement, correlation similarity,
contingency similarity + codes paths). The four kernels match the `methodology.md` formulas,
are in [0,1] with correct orientation, degrade to NaN (not crash) on degenerate inputs, use
Pearson as specified, and the codes/series paths agree to machine epsilon on the tested cases.
**One genuine correctness bug** found (repro-confirmed by me).

### B-1 (MEDIUM, NOTED) — `contingency_similarity` silently drops a legitimate `"nan"`-string category, making Trend understate divergence and disagree with Shape/Conditional
- **Where:** `metrics/primitives.py` `_as_labels` (~:169, `astype(str)`) + `_joint_freq` (~:288-295,
  the `!= "nan"` row filter).
- **What:** `_as_labels` stringifies categoricals with `astype(str)`, turning real `NaN` into the
  string `"nan"`. `_joint_freq` then drops rows where the label `== "nan"`. This correctly drops
  real NaN, but **also drops a legitimate category whose value is the exact lowercase string
  `"nan"`** — independently per side, so a real↔gen frequency mismatch on that category vanishes
  from the TV sum.
- **Why it matters:** the user-facing **Trend** metric calls `contingency_similarity` directly on
  raw user Series (`fidelity.py:112`). For a column with a genuine lowercase `"nan"` category,
  Trend overstates similarity, while **Shape** (`tv_complement`, `fidelity.py:78`) and **Conditional
  Trend** (`contingency_similarity_codes`) score the *same* data correctly — so one report carries
  mutually inconsistent numbers. The three kernels are inconsistent: TV/codes drop on the real
  `isna()` mask, contingency drops on a post-stringify string compare.
- **Blast radius (verified by me):** narrow trigger — only the exact lowercase `"nan"` string.
  Capital `"NaN"` is correctly kept (scores 0.5); real `np.nan` dropping is intended. Repro:
  real=50% `"nan"`+50% `"x"`, gen=all `"x"` → `contingency_similarity` returns **1.0**, but
  `tv_complement` and the codes path both return the correct **0.5**; a control category `"y"`
  with the same mode shift correctly returns 0.5. So only a genuine lowercase-`"nan"` category
  is mishandled — real but niche.
- **Fix:** drop on the original `isna()` mask (mirror `factorize_shared`) instead of the
  post-stringify `== "nan"` compare, so all three kernels treat a literal `"nan"` category
  identically and only real missingness is dropped. Closes the Series/codes divergence (also a
  blind spot in the vectorized-equivalence test, whose generator never emits the string `"nan"`).

---

## Unit C — metrics (conditional, diversity+encoding, fidelity, sanity)

Reviewed in 3 parallel passes. **C1 (conditional + base)** and **C3 (fidelity + sanity)**:
no statistical-correctness bugs — equal-per-slice aggregation, reference-only slice alignment,
NaN-safe column/pair means, and sanity denominators/orientations all verified correct. Two
recorded items below, both from **C2 (diversity + encoding)**.

> Note: a review agent initially flagged C-1 as a MEDIUM "~0.06 density discrepancy." On my own
> verification that magnitude was a **repro artifact** (the agent compared the live path against a
> dense call on freshly-regenerated random data). The controlled, same-data discrepancy is
> ~0.001 (float noise). Severity corrected to LOW. Kept here as a record of the real-but-tiny issue.

### C-1 (LOW, DONE) — dense vs structured PRDC `density` differ by float-noise (~0.001), contradicting the "identical PRDC" docstring
- **Resolution:** superseded by CPU-1 below; both Python representations now use shared float64 squared-distance kernels and strict membership, with dense/compact boundary and wide-feature equivalence tests. The details below record the original finding.
- **Where:** `metrics/diversity.py` — `compute_prdc` (dense, compares `sqrt(d) < sqrt(radius)` via
  `pairwise_distances`/`_kth_nn_distance`) vs `compute_prdc_structured` (compares `d² < radius²`).
  Dispatch at `_STRUCTURED_MIN_DIM = 512` (diversity.py:236).
- **What:** the two paths are equal in exact arithmetic, but `sqrt`-then-compare vs squared-compare
  round differently *exactly at tied distances*. Precision/recall/coverage are `any`-based and
  immune; **density** sums per-point boundary counts, so a few ties flip.
- **Verified magnitude (me):** categorical-only data → Δdensity = **0** (integer distances, no
  rounding gap, 16 seeds). Mixed numeric+categorical → max Δdensity ≈ **0.0013** over 19 seeds
  (~0.008% of a ~17 density). So the discrepancy is real but negligible — NOT the ~0.06 first
  reported.
- **Why it matters:** the docstrings (diversity.py:44, 67-69) and failure-modes.md promise the
  structured path yields "the same distances (hence the same PRDC)." For `density` in mixed data
  that's very slightly overstated, and the equivalence test (`test_prdc_structured_matches_dense`)
  only runs at onehot_dim=36 (below the 512 dispatch) with no tie-heavy case, so it can't catch it.
- **Fix (low priority):** make the dense path compare in squared space too (`d² < radius²`) so the
  two are provably bit-identical — OR soften the docstring to "identical up to float rounding in the
  density tie-count" and widen the equivalence test to run at >512 dim. Leaning: squared-space dense
  compare (eliminates it outright, cheap).

### C-2 (LOW, NOTED) — `fit_encoder` fills numeric NaN→0.0 *before* computing min/max, corrupting the fitted range
- **Where:** `encoding/features.py` `fit_encoder` / `_scaled_numeric` (~:78-84, 119-123).
- **What:** `apply(pd.to_numeric).fillna(0.0)` runs before `min`/`max`, so a NaN becomes a
  fabricated 0.0 that can widen the fitted range and shift the scaling.
- **Why it matters:** **production path is shielded** — `PRDC.compute` calls `dropna()` (diversity.py:231-232)
  before `fit_encoder`, so no NaN reaches it there. But `build_feature_pair` (a public `encoding`
  entry point, used by `test_calibration.py`) does NOT dropna, so any future caller gets a polluted
  range and wrong distances. Repro: `fit_encoder({"a":[10,11,12,NaN]})` → fitted min=0.0, range=12
  (true 10/12), scaling the valid values wrong.
- **Fix:** compute min/max ignoring NaN (drop or mask) before scaling, so the encoder's contract
  ("the real distribution defines the space") holds regardless of caller.

### C-3 (OPEN DESIGN QUESTION, needs author) — categorical vs numeric distance commensurability in the PRDC feature space
- **Where:** `encoding/features.py` (one-hot + range-scale feature space) feeding `metrics/diversity.py` (PRDC).
- **What:** a single one-hot category flip contributes Euclidean distance √2 to a row-pair, while a
  numeric column spanning its *entire* reference range contributes only 1.0 (range-scaled to [0,1]).
  So one categorical disagreement weighs ~1.4× a full-range numeric disagreement, and a row with many
  categorical columns is distance-dominated by them — which directly shapes the k-NN manifold radii and
  hence all four PRDC stats.
- **Status:** NOT a bug. This is standard one-hot + min-max PRDC, and the encoding is documented. But
  neither `methodology.md` nor any test asserts an *intended* relative weighting, so the categorical
  dominance is implicit rather than chosen.
- **Decision needed:** is the categorical dominance intentional (accept + document the weighting as a
  deliberate choice), or should one-hot blocks be rescaled (e.g. ×1/√2 per column, or 1/√p_cat) to be
  commensurate with the [0,1] numerics? Either way, pin the intended weighting with a doc note + a test
  so it can't drift silently.

---

## Historical Rust CPU migration follow-up

This section records earlier review evidence. The Rust backend has since been
removed; its comparisons remain historical and do not describe current execution.

### CPU-1 (DONE) — align Python PRDC with Rust's stable boundary comparisons
- **Status:** completed following the user's request to share PRDC/DCR distance code and align Python precision. `encoding/distances.py` retains accelerated scikit-learn dense neighbor selection and matrix products, refines cancellation/uncertain ranks/strict boundaries directly, and supplies common float64 distance tiles; encoding and metric-specific estimation are retained.
- **Where:** `metrics/diversity.py` — dense and structured Python PRDC paths.
- **What:** use float64 squared distances for both neighbor radii and cross-distances,
  retain strict `<` membership, and match Rust's direct recomputation near ambiguous
  boundaries while preserving efficient kernels and bounded memory.
- **Evidence:** identical sets of eight distinct categorical levels with `nearest_k=2`
  give Python density **4.0**, versus the correct strict-boundary density **0.5** in Rust.
  The Python radius is float64 `sqrt(2)` while its cross-distance is rounded to float32.
  This extends C-1: the earlier small discrepancies do not bound tie-heavy errors.
- **Validation:** the eight-level regression now gives density 0.5 on both backends. Independent direct-distance checks cover duplicates, unknown levels, dense/compact features beyond 512 coordinates, large-offset cancellation, tiny tile budgets, and stored bootstrap replicates. Full test lanes and before/after performance probes accompany the correction.

---

## Unit D — bootstrap / estimation engine

Reviewed: `bootstrap/{engine,resample,ci,flags}.py` + `stats.py`. **No correctness bugs.** The
statistical heart is sound — the three crown jewels were verified by repro (and I independently
re-confirmed the CI arithmetic):
- **CI recentering** — `CI = percentile(reps, α/2, 1−α/2) + (value − median)` is a width-preserving
  shift; algebraically identical to recentering the residuals before taking percentiles. Verified the
  point always sits inside its own CI (width 0.8550 preserved across shifts); Monte-Carlo coverage
  ≈93% at nominal 95% (expected mild percentile-bootstrap undercoverage, not a construction error).
- **Observed/ideal symmetry** — both paths use the same `size_policy`, `_point_estimate`, `_bootstrap`,
  metric, and `summarize_replicates`; only the inputs differ (value: holdout-vs-gen; ideal: two
  holdout halves). Ideal CI gated on the same reliability rule.
- **Holdout-split disjointness** — default split takes consecutive non-overlapping permutation slices;
  `ref+cand ≤ n_holdout` always, verified disjoint across odd/even/tiny n.
- **Determinism** — bit-identical across `n_jobs=1` vs `4` (seed-per-replicate via `SeedSequence.spawn`).

Items (all low/no-action):

### D-1 (NIT, OPTIONAL) — redundant clause in `_category_mismatch`
- **Where:** `bootstrap/engine.py:412` — `if t != s and (s - t or t - s)`.
- **What:** for sets, `t != s` already equals `(s−t) or (t−s)` non-empty; the second clause is dead
  logic. Correct, just removable. No behavior change.

### D-2 (NIT, OPTIONAL) — unresolved `ToDo` on `size_fraction` shipped in the statistical core
- **Where:** `bootstrap/resample.py:38-40`.
- **What:** a `ToDo` ("decide default size_fraction; same or different for reference vs candidate?")
  sits in the size-policy fn. Current behavior (one `size_fraction` floor-scales both sides, default
  1.0) is correct and test-covered. Resolve or remove the comment. **Open design question for author:**
  should reference and candidate ever take *different* size fractions, or is one-for-both final?

### D-3 (NIT, OPTIONAL) — flag precedence is undocumented
- **Where:** `bootstrap/engine.py:258 vs :264`.
- **What:** when a stat is both `not_applicable` and `insufficient`, only `NOT_APPLICABLE` is emitted
  (checked first). Reasonable, but the precedence isn't stated anywhere. Worth a one-line doc note.

---

## Unit E — cleaning

Reviewed: `cleaning/{__init__,coerce,nan_policy,normalize}.py`. **No correctness bugs.** The two
highest-risk properties hold: all transforms are pure value-wise functions with no data-fitted
parameters, so the holdout and generated frames are cleaned identically by construction; and
`clean()` does not mutate caller inputs (verified byte-identical after the call). Modes
(safe/off/strict) behave per `usage.md`, cleaning is idempotent, and the log accurately reports
column/count/action. Only minor edge cases:

### E-1 (LOW, NOTED) — `<MISSING>` sentinel collides with a real category literally named `<MISSING>`
- **Where:** `cleaning/nan_policy.py:25` + `cleaning/__init__.py:79-84`, under `categorical_nan_policy="category"`.
- **What:** filled NaNs and any pre-existing real `<MISSING>` values become indistinguishable, and the
  log counts only the filled NaNs (undercount). Repro: holdout `["a",None,"<MISSING>",None]` → cleaned
  has 3 `<MISSING>` but log says "labelled 2 NaN".
- **Why it's LOW:** requires a real category named exactly `<MISSING>`, and the merge is symmetric across
  both frames so the comparison isn't corrupted — just the missingness semantics and the log count.
- **Fix (if desired):** uniqueness-check the sentinel (suffix on collision) or log a warning when the label
  already exists in the column.

### E-2 (NIT, OPTIONAL) — datetime cast logged unconditionally (numeric/categorical log only on change)
- **Where:** `cleaning/coerce.py:69-72`.
- **What:** datetime→epoch always appends a log action even for an already-clean datetime column, unlike
  the numeric/categorical branches which gate on `.equals()`. Defensible (the cast does change
  representation), but it means the log is never empty when any datetime column is present. Minor
  consistency call. (Also NIT: `nan_policy` inf-replacement redundantly re-touches already-cast datetime
  epoch columns — harmless.)

---

## Unit F — registry + report / output

Reviewed: `registry/registry.py`, `report/{result,io}.py`, `stats.py` (output-facing). All output
views iterate the registry (no hard-coded metric names), JSON is deterministic/byte-stable, flags
surface in df/json/markdown, and value/CI/ideal columns map correctly. **One real HIGH bug.**

### F-1 (HIGH, NOTED) — console `summary()` silently drops flags of omitted (all-NaN) categories, hiding blocking unreliability
- **Where:** `report/io.py:182-199` — `used_flags` is accumulated *inside* the per-category loop, AFTER
  the all-NaN `continue` at :188. The markdown view (:134-135) accumulates over ALL stats, so the two
  views disagree.
- **What (verified by me):** when small `n_eff` triggers `INSUFFICIENT_DATA` on every non-sanity metric,
  those whole categories are omitted from the summary, so their flag codes never enter the legend. Result:
  `summary()` shows ONLY the Sanity block — no metric rows, no `flags (...)` legend, and `INSUFFICIENT_DATA`
  appears nowhere. `to_dataframe()` and `to_markdown()` both DO surface it. Repro: `evaluate(df, df, config=
  EvalConfig(n_bootstrap=40, min_bootstrap_n=1000))` on n=60 → `"INSUFFICIENT_DATA" in summary()` is False;
  in markdown/dataframe it's True.
- **Why it matters:** directly violates the package's core guarantee — `flags.py:5` "No flag is ever silent
  — renderers print a legend"; README "never reports a number without the context to trust it." In the
  PRIMARY console view, a user with too-little data sees their metrics silently vanish with no explanation.
  Numbers aren't wrong, but the unreliability is hidden exactly where it's most likely to mislead.
- **Why the test misses it:** `test_summary_omits_allnan_and_has_legend` uses a fixture where fidelity
  renders, so it only checks that *listed* flags are described — never that an omitted category's flags
  still reach the legend.
- **Fix:** compute `used_flags` over all stats (as markdown does), independent of category omission — and
  consider a one-line "N metrics omitted (INSUFFICIENT_DATA): see table" placeholder so the console view
  doesn't look like only sanity ran.

### F-2 (LOW, NOTED) — "round-trippable" JSON has no deserializer; `display`/`fmt` dropped
- **Where:** `README.md:84` + `report/io.py:87` docstring claim "round-trippable"; no `from_json`/`load`
  exists on `Report`, and the JSON omits `display`/`fmt` and encodes NaN as `null` (indistinguishable from a
  genuine `None`). The emitted JSON is valid and lossless for the fields it includes (that's all
  `test_to_json_roundtrips` asserts), but a `Report` can't be reconstructed. Doc-precision, not data loss.
- **Fix:** either add a `Report.from_json` (and include `display`/`fmt`), or reword the claim to "valid,
  re-loadable JSON."

### F-3 (NIT) — `_fmt` renders `±inf` as `"NaN"` (report/io.py:225). Effectively unreachable (cleaning maps inf→NaN upstream); cosmetic.
### F-4 (NIT) — header provenance fields (`n_holdout`/`n_generated`/`n_bootstrap`) rendered without `.get` defaults (report/io.py:177-179); `evaluate()` always populates them, so robustness-only.

---

## Unit G — orchestration + entry (evaluate / cli / io_files / datasets)

Reviewed: `evaluate.py`, `cli.py`, `io_files.py`, `datasets/openml.py`. **No statistical-correctness
bugs.** The crown jewel — `load_openml`'s `(train, holdout)` split — is deterministic, leakage-free
(disjoint partition covering all rows), correctly fractioned, index-reset (verified offline via a
monkeypatched fetch). `evaluate()` pipeline order matches `architecture.md` (type-check → infer-from-
holdout → validate → clean → size-policy → warn → run), schema is inferred from the HOLDOUT, and
end-to-end determinism holds (serial == `n_jobs=4`). Two CLI ergonomics items:

### G-1 (MEDIUM, NOTED) — CLI surfaces `ValidationError`/`FileNotFoundError` as a raw traceback
- **Where:** `cli.py:108` (and `:89`) — `main()` calls `read_table()`/`evaluate()` with no try/except.
- **What:** misuse (bad path, mismatched columns, unsupported dtype) propagates as an uncaught exception —
  exit code is correctly non-zero (1), but the actionable message is buried under a 20-line traceback. The
  README frames the CLI as the user-facing on-ramp and promises misuse "raises with an actionable message."
- **Fix:** catch `ValidationError`/`FileNotFoundError` in `main()` and print a one-line `error: ...` to
  stderr (keep the non-zero exit). **Open question:** is a traceback intended, or a clean message?

### G-2 (LOW, NOTED) — `--metrics` with the flag present but no names silently yields an empty report
- **Where:** `cli.py:23` (`nargs="*"`) + `:108`.
- **What:** `ltm-eval --metrics` → `args.metrics == []` → `resolve_metrics([])` returns ZERO metrics →
  Report with no metric rows, exit 0. Omitting the flag gives `None` → all default metrics. A user who types
  `--metrics` expecting "all" gets a silently empty evaluation with a success code.
- **Fix:** treat empty list as `None` (= all), or reject it with an error.

### G-3 (NIT) — `read_table` silently falls back to CSV for unknown extensions, and `--out` routes any
non-`.json` to markdown (`io_files.py:16-17`, `cli.py:111-114`). Defensible defaults; don't match a strict
reading of usage.md's "CSV/Parquet" / "`.md`/`.json` by extension". No statistical impact.
