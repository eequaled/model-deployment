"""Shared fixtures: a tiny model directory and an engine built from it."""

from __future__ import annotations

import pytest

from server.ocr_engine import FlorenceOCR
from tests.tinymodel import build_tiny_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny-model"))


@pytest.fixture(scope="session")
def engine(tiny_model_dir) -> FlorenceOCR:
    return FlorenceOCR(tiny_model_dir, threads=2)
