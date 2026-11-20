import pytest

from app import db, media, projects
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


@pytest.fixture
def street(conn, image):
    """A project: a small car; a big car and a person; nothing."""
    objects = {
        "small.png": [{"class": "car", "bbox": [5, 5, 10, 10]}],
        "big.png": [{"class": "car", "bbox": [5, 5, 40, 40]}, {"class": "person", "bbox": [0, 0, 8, 20]}],
        "empty.png": [],
    }
    with db.action("request", "make"), conn:
        project_id = projects.create(conn, "Street")
        for n, (name, found) in enumerate(objects.items()):
            media_id, _ = media.store(conn, project_id, image(n, size=(64, 64)), name)
            projects.save(conn, media_id, project_id, found)
    return project_id


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
