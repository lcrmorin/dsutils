**Repo:** pandas-dev/pandas
**Type:** Feature request (ENH)

## Title
Styler: built-in way to render several styled DataFrames side by side

## Is your feature request related to a problem?

Showing two or three small styled tables next to each other (for comparison - before/after, train/test, group A/B) is a common notebook need with no built-in support. The standard workaround, independently reinvented across many blog posts, Stack Overflow answers, and internal notebooks, is:

```python
html = [df.style.set_table_attributes("style='display:inline'").set_caption(title)._repr_html_()
        for df, title in zip(df_list, title_list)]
display_html("\xa0\xa0\xa0".join(html), raw=True)
```

This reaches into `Styler._repr_html_()` (a private-by-convention method) and hand-builds raw HTML with manual spacing - fragile against internal `Styler` HTML changes, and something this many people are independently re-deriving is a sign it belongs in the library.

Checked whether an existing pandas Styler feature already covers this: `Styler.concat()` (added in pandas 1.5) combines two Stylers, but *vertically* (row-wise - e.g. appending a totals row), not side by side. `itables` was suggested to me as an alternative elsewhere, but it renders one interactive DataTable per call, block-level, with no side-by-side layout option and no support for styled/`Styler` input at all - a different tool for a different job, not a substitute.

## Describe the solution you'd like

Something like `pandas.io.formats.style.Styler.display_side_by_side([styler1, styler2, ...], captions=[...])`, or a module-level `pandas.io.formats.style.concat_horizontal(...)`, producing the equivalent inline HTML without touching `_repr_html_()` directly.

## Describe alternatives you've considered

Keeping this as a small user-level helper function (which is what I've done) works fine - filing mainly because the workaround is common enough, and reaches deep enough into non-public API, that it seems worth a supported answer rather than everyone re-deriving the same HTML-concatenation trick.
