from psy_experiment.instruments import load_instrument
from psy_experiment.models import Language


def test_loads_32_items_with_ids_1_to_32():
    items = load_instrument()
    assert len(items) == 32
    assert [i.id for i in items] == list(range(1, 33))


def test_q1_english_poles_match_oejts():
    items = {i.id: i for i in load_instrument()}
    left, right = items[1].left_en.lower(), items[1].right_en.lower()
    assert "list" in left
    assert "memory" in right


def test_chinese_poles_present():
    item = load_instrument()[0]
    assert item.left_zh.strip()
    assert item.right_zh.strip()
