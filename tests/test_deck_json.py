"""Guard against pydeck serializing a literal string prop as an accessor expression.

pydeck turns a bare string into an "@@=<expr>" accessor, which deck.gl parses as JavaScript in
the browser. A literal (an image data URI, a font family, a unit name) has to be wrapped in
pydeck.types.String or the map dies in the browser while every headless test still passes.
"""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ACCESSOR = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")     # a plain field name like "pos" or "color"


def accessor_strings(node, out):
    if isinstance(node, dict):
        for v in node.values():
            accessor_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            accessor_strings(v, out)
    elif isinstance(node, str) and node.startswith("@@="):
        out.append(node[3:])
    return out


@pytest.mark.parametrize("script", ["app.py", "app_brigade.py"])
def test_no_literal_is_sent_as_an_accessor(script):
    at = AppTest.from_file(str(ROOT / script), default_timeout=300).run()
    assert not at.exception
    charts = at.get("deck_gl_json_chart")
    assert charts, "no pydeck chart rendered"
    for chart in charts:
        spec = json.loads(chart.proto.json)
        for expr in accessor_strings(spec, []):
            assert ACCESSOR.match(expr), (
                f"{script}: pydeck sent the literal {expr!r} as an accessor; "
                "wrap it in pydeck.types.String")
