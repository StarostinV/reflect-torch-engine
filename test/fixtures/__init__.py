import pytest

from test.fixtures.get_data import get_test_data

TEST_DATA = list(get_test_data())


@pytest.fixture(params=TEST_DATA)
def reflectivity_data(request):
    return request.param
