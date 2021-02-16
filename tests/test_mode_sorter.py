#!/usr/bin/env python

"""Tests for `mode_sorter` package."""

import pytest


from mode_sorter import mode_sorter
import numpy as np


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_transfer_coefficient():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    assert mode_sorter.transfer_coefficient(x, y, 1) == 2
