import shutil
import pytest

from tests.utils import get_temp_output_path

@pytest.fixture
def temp_dir():
    dir_path = get_temp_output_path()

    # delete temp folder if it existent (clean startup)
    if dir_path.exists():
        shutil.rmtree(dir_path)

    # create new temp folder
    print(f"Creating {dir_path}")
    dir_path.mkdir(exist_ok=True)

    # provide directory for test runs
    yield dir_path

    # teardown (delete temp folder after test)
    if dir_path.exists():
        print(f"Deleting {dir_path}")
        shutil.rmtree(dir_path)
