import pytest


def pytest_addoption(parser):
    parser.addoption("--run-weights", action="store_true", help="Enable real-weight tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "weights: loads the real TinyLlama model weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-weights"):
        return
    skip_weights = pytest.mark.skip(reason="real weights require --run-weights")
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip_weights)
