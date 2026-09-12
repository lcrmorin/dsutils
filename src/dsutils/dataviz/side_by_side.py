"""Display several styled DataFrames side by side in a notebook.

Kept as its own function rather than replaced by itables - verified
itables.show() renders one interactive DataTable per call, block-level, with
no inline/side-by-side layout option, and it doesn't accept `.style` objects
the way this does. Different tool for a different job (see
existing_package_mapping.md).

Only real change from the original: explicit `from IPython.display import
display_html` instead of relying on it being injected into a notebook's
global namespace, so this works from a plain script/import too (in a
non-notebook environment `raw=True` HTML just won't render, but the function
won't NameError).
"""

from IPython.display import display_html


def display_sbs(df_list, title_list):
    """Render a list of styled DataFrames inline, side by side, each under
    its own caption.

    df_list: list of pandas Styler objects (e.g. `df.style`), not plain
        DataFrames - `.set_table_attributes`/`.set_caption` are Styler methods
    title_list: list of caption strings, same length as df_list
    """
    html = [
        df.set_table_attributes("style='display:inline'").set_caption(title)._repr_html_()
        for df, title in zip(df_list, title_list)
    ]
    display_html("\xa0\xa0\xa0".join(html), raw=True)
