# Reference snippets (not part of the installable `dsutils` package)

Every viz-related thing now lives under `dataviz/`: real functions (`percentogram.py`, `fold_sizes.py`) one level up, and these three reference-only scripts here. They're kept as-is rather than turned into package functions, because they aren't utilities - they're a few lines of direct `seaborn`/`matplotlib_venn` calls with your preferred default arguments baked in (jitter amounts, color maps, figure titling). There's nothing to wrap: calling `sns.pairplot(...)` or `venn3(...)` yourself is exactly as short as calling a function that calls it for you, and a real function would just be hiding the one line that matters.

Because this folder has no `__init__.py`, it isn't a Python subpackage and won't be picked up by `pip install` (setuptools only ships importable packages by default) - that's intentional, these aren't meant to be imported. They live here for the repo's sake, not the wheel's.

They also reference undefined globals (`df`, `feature`, `target`, `all_data`) rather than taking parameters - copy the relevant lines into your notebook and fill in your own variable names, rather than importing these.

- `data_viz.py` - styled `.background_gradient()` table + a jittered regression pairplot recipe.
- `term_structure_viz.py` - the same jittered pairplot recipe, applied to a `targets` list.
- `data_viz_venn.py` - a 3-set Venn diagram via `matplotlib_venn.venn3` (still has the original's unfinished `# TODO: automatically calculate subsets` - subset sizes are hardcoded, you'd need to compute them yourself for real data).
