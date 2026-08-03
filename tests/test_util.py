"""The shared helpers behave identically to the copies they replaced."""

from file_index import util
from file_index.extractors import audio as audio_ex
from file_index.index import fts_escape as index_fts_escape
from file_index.search import format_ts as search_format_ts
from file_index.web import _fts_escape as web_fts_escape


def test_single_definition_is_reused_everywhere():
    assert index_fts_escape is util.fts_escape
    assert web_fts_escape is util.fts_escape
    assert search_format_ts is util.format_ts


def test_fts_escape_neutralizes_operators():
    assert util.fts_escape('cat AND "dog" OR (x)') == '"cat" "AND" """dog""" "OR" "(x)"'
    assert util.fts_escape("  spaced   out  ") == '"spaced" "out"'
    assert util.fts_escape("") == ""


def test_strip_think_removes_blocks():
    assert util.strip_think("<think>hmm</think>answer") == "answer"
    assert util.strip_think("<think>a\nb</think>  x  ") == "x"
    assert util.strip_think("no blocks") == "no blocks"


def test_format_ts_matches_transcript_formatting():
    assert util.format_ts(None) == ""
    assert util.format_ts(0) == "00:00"
    assert util.format_ts(75) == "01:15"
    assert util.format_ts(3725) == "01:02:05"
    # format_transcript renders through the same helper
    line = audio_ex.format_transcript([{"start": 75, "end": 3725, "text": "hi"}])
    assert line == "[01:15 - 01:02:05] hi"
