from source.cnn.projects.output.training_records import TrainingRecords

from os import remove
from os.path import dirname, realpath, exists
from tinydb import TinyDB, Query
from datetime import datetime

import pytest


@pytest.fixture(scope="module", autouse=True)
def record_dir():
    return dirname(realpath(__file__))


@pytest.fixture(scope="module", autouse=True)
def delete_record_file(record_dir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    path = f"{record_dir}/training_record.json"
    if exists(path):
        remove(path)


def test_load_db(record_dir):
    record = TrainingRecords(record_dir)

    assert exists(record.db_file)
    assert type(record.db) == TinyDB


def test_save_training_record(record_dir):
    record = TrainingRecords(record_dir)

    count = record.save("20231009202121", "vgg16", "a")

    assert count == 1

    count = record.save("20231009202121", "vgg16", "a")

    assert count == 1


def test_update_created_training_record(record_dir):
    record = TrainingRecords(record_dir)

    count = record.save("20231009202121", "vgg16", "a")

    assert count == 1

    count = record.save("20231009202121", "vgg16", "a", 0.84, "20231009212121")

    assert count == 1
