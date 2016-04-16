import logging
import os
from os.path import dirname

from playset.attributes import Purple, Green, Red, Solid, Shaded, Empty, Squiggle, Diamond, Oval, Card
from playset.identify import id_attributes, CardRepo

Truth = {
    0: Card(Purple, 3, Solid, Squiggle),
    1: Card(Green, 2, Solid, Diamond),
    2: Card(Purple, 2, Empty, Diamond),
    3: Card(Green, 3, Shaded, Squiggle),
    4: Card(Red, 2, Shaded, Diamond),
    5: Card(Green, 3, Empty, Oval),
    6: Card(Red, 1, Solid, Squiggle),
    8: Card(Green, 2, Empty, Oval),
    9: Card(Green,2,Shaded,Squiggle),
    10: Card(Red,1,Empty,Squiggle),
    11: Card(Purple,1,Shaded,Diamond),
    12: Card(Purple,1,Shaded,Squiggle),
    13: Card(Purple,2,Shaded,Squiggle),
    14: Card(Purple,3,Solid,Oval),
    15: Card(Green,1,Empty,Squiggle),
    16: Card(Purple,2,Empty,Oval),
    17: Card(Purple,1,Empty,Squiggle),
    18: Card(Green,2,Solid,Squiggle),
    19: Card(Red,3,Empty,Diamond),
    20: Card(Green,2,Shaded,Oval),
    21: Card(Purple,3,Empty,Diamond),
    22: Card(Red,1,Solid,Diamond),
    23: Card(Red,2,Shaded,Oval),
    24: Card(Purple,1,Solid,Oval),
    25: Card(Purple,1,Empty,Oval),
    26: Card(Green,1,Solid,Oval),
    27: Card(Green,3,Solid,Diamond),
    28: Card(Red,2,Empty,Diamond),
    29: Card(Red,3,Shaded,Squiggle),
    30: Card(Purple,1,Empty,Diamond),
    31: Card(Red,1,Shaded,Diamond),
    32: Card(Red,2,Empty,Squiggle),
    33: Card(Purple,3,Solid,Diamond),
    34: Card(Green,3,Shaded,Oval),
    35: Card(Red,2,Shaded,Squiggle),
    36: Card(Purple,1,Shaded,Oval),
    37: Card(Red,3,Shaded,Oval),
    38: Card(Red,2,Solid,Diamond),
}


def test_group():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(funcName)s %(levelname)s - %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    repo = get_repo()

    assert_card(0, repo)
    assert_card(1, repo)
    assert_card(2, repo)
    assert_card(3, repo)
    assert_card(4, repo)
    assert_card(5, repo)
    assert_card(6, repo)
    assert_card(8, repo)
    assert_card(9, repo)
    assert_card(10, repo)
    assert_card(11, repo)
    assert_card(12, repo)
    assert_card(13, repo)
    assert_card(14, repo)
    assert_card(15, repo)
    assert_card(16, repo)
    assert_card(17, repo)
    assert_card(18, repo)
    assert_card(19, repo)
    assert_card(20, repo)
    assert_card(21, repo)
    assert_card(22, repo)
    assert_card(23, repo)
    assert_card(24, repo)
    assert_card(25, repo)
    assert_card(26, repo)
    assert_card(27, repo)
    assert_card(28, repo)
    assert_card(29, repo)
    assert_card(30, repo)
    assert_card(31, repo)
    assert_card(32, repo)
    assert_card(33, repo)
    assert_card(34, repo)
    assert_card(35, repo)
    assert_card(36, repo)
    assert_card(37, repo)
    assert_card(38, repo)


def assert_card(i, repo):
    logging.debug(i)
    expected = Truth.get(i)
    c = repo.get(i)
    actual = id_attributes(c)
    assert (actual.count == expected.count)
    assert (actual.shape == expected.shape)
    assert (actual.color == expected.color)
    assert (actual.shading == expected.shading)


def get_repo():
    path = dirname(os.path.abspath(__file__))
    root = os.path.join(path, '..', 'data', 'training')
    repo = CardRepo(root)
    return repo
