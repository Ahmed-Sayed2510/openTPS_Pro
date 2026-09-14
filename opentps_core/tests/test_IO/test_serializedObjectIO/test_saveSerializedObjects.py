import pickle
from pathlib import Path
import bz2

import pytest

from opentps.core.io.serializedObjectIO import saveSerializedObjects

# Basic round-trip: write pickle file and load it back.
def test_saveSerializedObjects_pickle_ok(tmp_path: Path) -> None:
    saving_path = tmp_path / "data"
    data = [1, 2, 3]

    saveSerializedObjects(data, str(saving_path), compressedBool=False, dictionarized=False)
## create fixture 
    output_file = Path(str(saving_path) + ".p")
    assert output_file.is_file()

    with output_file.open("rb") as f:
        loaded = pickle.load(f)

    assert loaded == data
    
# Compressed mode: ensure pbz2 file is created and can be loaded.
def test_saveSerializedObjects_compressed_ok(tmp_path: Path) -> None:
    saving_path = tmp_path / "data"
    data = [1, 2, 3]

    saveSerializedObjects(data, str(saving_path), compressedBool=True, dictionarized=False)

    output_file = Path(str(saving_path) + "_compressed.pbz2")
    assert output_file.is_file()

    with bz2.BZ2File(output_file, "rb") as f:
        loaded = pickle.load(f)

    assert loaded == data
    
# Non-list input should be wrapped into a single-item list.
def test_saveSerializedObjects_wraps_non_list(tmp_path: Path) -> None:
    saving_path = tmp_path / "data"
    data = 42

    saveSerializedObjects(data, str(saving_path), compressedBool=False, dictionarized=False)

    output_file = Path(str(saving_path) + ".p")
    with output_file.open("rb") as f:
        loaded = pickle.load(f)

    assert loaded == [42]
    
# Dictionarized mode should apply dictionarizeData to each element before saving.
def test_saveSerializedObjects_dictionarized_ok(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_dictionarize(obj):
        calls.append(obj)
        return {"value": obj}

    monkeypatch.setattr(
        "opentps.core.io.serializedObjectIO.dictionarizeData",
        fake_dictionarize,
    )

    saving_path = tmp_path / "data"
    data = ["a", "b"]

    saveSerializedObjects(data, str(saving_path), compressedBool=False, dictionarized=True)

    output_file = Path(str(saving_path) + ".p")
    with output_file.open("rb") as f:
        loaded = pickle.load(f)

    assert calls == ["a", "b"]
    assert loaded == [{"value": "a"}, {"value": "b"}]
    
# Invalid output path should raise an exception.
def test_saveSerializedObjects_invalid_path_raises(tmp_path: Path) -> None:
    saving_path = tmp_path / "missing_dir" / "data"

    with pytest.raises(FileNotFoundError):
        saveSerializedObjects([1, 2, 3], str(saving_path), compressedBool=False, dictionarized=False)



