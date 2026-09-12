**Repo:** pandas-dev/pandas
**Type:** Documentation gap (tie to #59328)

## Title
String dtype migration guide: explicitly call out "branch on `dtype != object` to mean numeric" as a breaking pattern

## Describe the bug / documentation gap

#59328 already tracks the broad category "code checking for the dtype (e.g. `ser.dtype == object`) assuming object dtype, will break" under the pandas 3.0 default-string-dtype change - but one specific, very common instantiation of that pattern isn't called out explicitly anywhere I could find (the tracking issue, the migration guide): code that uses `dtype != object` (or `dtype == object`, inverted) as a proxy for "is this column numeric," then proceeds to call numeric-only operations in the non-object branch.

This is a common pattern in "auto-detect column types and process accordingly" utilities (dtype-based dispatch loops, memory-optimization helpers, auto-EDA tools) - reasonable code on any pandas version before 3.0, where "not object" and "is numeric" were effectively synonymous for the column types such code usually saw. On pandas >= 3.0 with the new default string dtype, a plain string column is no longer `object`, so it silently falls into the "numeric" branch and crashes on the first numeric-only call (`.min()`, `np.finfo(...).min` comparisons, etc.) - not a deprecation warning, a hard `TypeError`/`UFuncTypeError` at the first numeric op.

## Reproducible example

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"s": ["a", "b", "c"]})
col_type = df["s"].dtype
print(col_type, col_type != "object")  # 'str' True  <- silently treated as "not object"

if col_type != "object":
    df["s"].min()  # UFuncTypeError comparing strings inside a numeric-typed branch downstream
```

## Expected behavior

Not asking for a behavior change here (the dtype change itself is intentional and already documented) - asking that the migration guide's list of breaking patterns explicitly names "branching on `!= object`/`== object` to select a numeric-vs-non-numeric code path" as its own bullet, with the recommended fix (`pandas.api.types.is_numeric_dtype()` / `is_string_dtype()` instead of a literal dtype-name comparison), since it's a distinct enough failure mode from the ones already listed (astype(str) NaN handling, mixed-dtype assignment, sum/prod on string dtype) that someone skimming the existing list could easily miss that their code has this specific problem.

## Installed versions

pandas 3.0.2, Python 3.11, Linux.
