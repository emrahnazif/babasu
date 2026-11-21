import pytest

from app import projects
from app.filters import Filter, FilterError


@pytest.mark.parametrize("text", [
    "labelled name:night objects:>20 (car w:8..16 h:8..16) -(person)",
    "unlabelled car|truck -person",
    '"traffic light" (car|"fire truck" size:<=16 edge:no)',
    'name:"street 2" objects:3 (area:100.. aspect:..0.5 edge)',
    "(w:>=8 h:<4.5)",
    "-(edge)",
])
def test_the_text_form_reads_back_as_the_same_filter(text):
    """What URLs carry and the agent writes: written out and read again, a filter is the same."""
    parsed = Filter.parse(text)
    assert Filter.parse(parsed.text()) == parsed


@pytest.mark.parametrize("text", ["(car", "car)", 'name:"night', "(w:8-16)", "objects:..", "width:8", "(colour:red)"])
def test_text_that_cant_be_read_is_refused(text):
    with pytest.raises(FilterError):
        Filter.parse(text)


@pytest.mark.parametrize("text, names", [
    ("car", {"small.png", "big.png"}),
    ("(car size:8..16)", {"small.png"}),
    ("(car w:>10)", {"big.png"}),  # > leaves the end out
    ("-person", {"small.png", "empty.png"}),
    ("unlabelled", {"empty.png"}),
    ("objects:>1", {"big.png"}),
    ("(person edge)", {"big.png"}),
    ("name:SMA", {"small.png"}),
    ("CAR -(car size:8..16)", {"big.png"}),  # class names in any case
])
def test_a_filter_finds_the_images_it_describes(conn, street, text, names):
    found = projects.browse(conn, Filter.parse(text), project_id=street)["images"]
    assert {i["filename"] for i in found} == names


def test_a_class_the_project_doesnt_have_is_refused(conn, street):
    """A typo would otherwise find nothing, which looks like an answer."""
    with pytest.raises(FilterError):
        projects.browse(conn, Filter.parse("cra"), project_id=street)


@pytest.mark.parametrize("text, says", [
    ("(car w:<10) | (person)", "“|” can't join them"),  # it read as an empty term, matching nothing
    ("car w:<10", r"goes in brackets with their class: \(car w:<10\), or \(w:<10\)"),
    ("edge:yes", "goes in brackets"),
    ("class:car", "a class by its name"),
    ("colour:red", "Terms: labelled or unlabelled"),
])
def test_a_term_that_isnt_one_says_how_to_write_it(text, says):
    with pytest.raises(FilterError, match=says):
        Filter.parse(text)


def test_statistics_of_a_class_say_how_to_count_its_images(conn, street):
    with pytest.raises(FilterError, match=rf"GET /api/images\?project_id={street}&where=car"):
        projects.stats(conn, street, "car")


def test_deleting_objects_without_naming_them_says_how(conn, street):
    with pytest.raises(ValueError, match=r"\(car\) for every car"):
        projects.change(conn, street, Filter(), "delete_objects", None, None)
