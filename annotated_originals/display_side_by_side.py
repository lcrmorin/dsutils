import pandas as pd

# Display dataframe stlyes side by side
def display_sbs(df_list, title_list):
    html = [df.set_table_attributes("style='display:inline'")
            .set_caption(title)._repr_html_()
            for df, title in zip(df_list,title_list)]

    display_html("\xa0\xa0\xa0".join(html), raw=True)

# ---------------------------------------------------------------------------
# itables (pip install itables) does NOT replace this - checked its actual
# `show()` signature and output, not just the pitch. `itables.show(df,
# caption=...)` renders ONE interactive DataTable per call, injected as its
# own block-level output; there's no argument to lay multiple tables out
# inline like this function's `display:inline` trick, and it doesn't take
# styled (`df.style`) objects the way this function does. itables solves a
# different problem (one big sortable/searchable/paginated table) than
# this one (several small static styled tables side by side for visual
# comparison).
#
# Keep this function if the side-by-side layout is the point. If you just
# want one nicer, interactive table, `itables.show(df, caption=title)` per
# table is fine - they'll stack vertically, not sit side by side.
# ---------------------------------------------------------------------------
