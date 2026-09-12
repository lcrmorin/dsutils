**Repo:** BayesWitnesses/m2cgen
**Type:** Feature request

## Title
Add SQL as a supported export language

## Is your feature request related to a problem?

m2cgen converts trained models into Java, C, Python, Go, JavaScript, Visual Basic, C#, R, PowerShell, PHP, Dart, Haskell, Ruby, F#, and Rust - a long list of general-purpose languages - but not SQL, even though "run this model inside the database, no application server involved" is one of the most common reasons people want a portable model export in the first place (scoring directly in a data warehouse query, a BI tool, or anywhere you can't deploy a Python/Java runtime next to the data).

Checked the closest existing options before filing this:
- `gbm2sql`, `sqlgbm`, `xgb2sql` - narrow, single-purpose, effectively unmaintained (1 commit / "not ready for production" in their own README / R-only).
- PMML export (`jpmml-xgboost` etc.) covers *some* of this, but requires a PMML-capable scoring engine on the other end - not plain SQL you can run in an arbitrary Postgres/MySQL/Snowflake/BigQuery instance.

So there's a real, currently-unfilled gap specifically for "arbitrary tree-based model -> plain SQL," and m2cgen already has the right shape (a well-tested multi-target code generator with tree-model support) to be the natural home for it.

## Describe the solution you'd like

A new output target, e.g. `m2cgen.export_to_sql(model, dialect="postgres")`, generating a `CASE WHEN ... THEN ... END` (or `IF`) expression tree per output, the same way the existing language backends walk the model's internal tree structure. Needs dialect awareness for at least: string-literal quoting, and each dialect's exact-cast syntax for matching a model's internal float precision (`CAST(x AS REAL)` in Postgres vs `CAST(x AS FLOAT)` in MySQL vs `CAST(x AS FLOAT64)` in BigQuery - these genuinely differ and a naive single CAST string doesn't work everywhere).

## Describe alternatives you've considered

Maintaining a standalone converter outside m2cgen (which is what I've done so far, XGBoost-only, hardened against several dtype/precision edge cases) - happy to contribute what I've learned about the tricky parts (float32 exact-literal matching, categorical split handling, base_score-to-margin conversion for non-identity link functions) if a SQL target is something the project wants, rather than duplicating the multi-model-type support m2cgen already has.

## Additional context

Have a working standalone XGBoost-to-SQL converter (maintained as its own project, not part of m2cgen) that's already hit and worked through the harder edge cases here - float32 exact-literal matching, categorical split handling, non-identity link functions, an engine-portability limitation on SQLite specifically (no true float32 CAST there). Happy to share specifics on any of that if it's useful background once/if a SQL target is on the table.
