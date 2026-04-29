"""
Unit tests for pure-Python functions in src/search.py.
No Azure credentials or network calls required.
"""

from src.search import _build_filter, _eq


# ---------------------------------------------------------------------------
# _eq — individual field=value OData expressions
# ---------------------------------------------------------------------------

def test_eq_string_wraps_in_single_quotes():
    assert _eq("category", "manual") == "category eq 'manual'"


def test_eq_string_escapes_single_quotes():
    assert _eq("name", "O'Brien") == "name eq 'O''Brien'"


def test_eq_int_no_quotes():
    assert _eq("page_number", 3) == "page_number eq 3"


def test_eq_bool_true():
    assert _eq("active", True) == "active eq true"


def test_eq_bool_false():
    assert _eq("active", False) == "active eq false"


# ---------------------------------------------------------------------------
# _build_filter — dict → OData $filter string
# ---------------------------------------------------------------------------

def test_build_filter_none_input():
    assert _build_filter(None) is None


def test_build_filter_empty_dict():
    assert _build_filter({}) is None


def test_build_filter_single_string_value():
    result = _build_filter({"category": "manual"})
    assert result == "category eq 'manual'"


def test_build_filter_single_int_value():
    result = _build_filter({"page_number": 5})
    assert result == "page_number eq 5"


def test_build_filter_list_of_strings_produces_or_clause():
    result = _build_filter({"category": ["manual", "policy"]})
    assert result is not None
    assert "category eq 'manual'" in result
    assert "category eq 'policy'" in result
    assert " or " in result
    # entire OR clause must be wrapped in parens
    assert result.startswith("(")
    assert result.endswith(")")


def test_build_filter_single_quote_escaped_in_value():
    result = _build_filter({"source_path": "manuals/O'Brien.pdf"})
    assert "O''Brien" in result


def test_build_filter_multiple_fields_joined_with_and():
    result = _build_filter({"category": "manual", "file_type": "pdf"})
    assert result is not None
    assert " and " in result
    assert "category eq 'manual'" in result
    assert "file_type eq 'pdf'" in result


def test_build_filter_bool_value():
    result = _build_filter({"active": True})
    assert result == "active eq true"
