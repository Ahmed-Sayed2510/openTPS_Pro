from types import SimpleNamespace
import pytest

# Import the module (not only the function) so that
# saveSerializedObjects can be replaced by a fake implementation
# inside the module namespace
import opentps.core.io.serializedObjectIO as io_mod


@pytest.fixture

def mock_saveSerializedObjects(monkeypatch):
    """
    Fixture that replaces saveSerializedObjects to avoid real disk I/O
    and records all calls for inspection.
    """
    calls = []

    def fake_saveSerializedObjects(data, path, compressedBool=False):
        calls.append((data, path, compressedBool))

    monkeypatch.setattr(io_mod, "saveSerializedObjects", fake_saveSerializedObjects)

    return calls


def test_saveDataStructure_split_false(mock_saveSerializedObjects):
    calls = mock_saveSerializedObjects

    p1 = SimpleNamespace(name="Alice")
    p2 = SimpleNamespace(name="Bob")
    patient_list = [p1, p2]

    io_mod.saveDataStructure(
        patientList=patient_list,
        savingPath="/tmp/patients",
        compressedBool=False,
        splitPatientsBool=False
    )

    assert len(calls) == 1
    assert calls[0][0] is patient_list
    assert calls[0][1] == "/tmp/patients"
    assert calls[0][2] is False


def test_saveDataStructure_split_true(mock_saveSerializedObjects):
    calls = mock_saveSerializedObjects

    p1 = SimpleNamespace(name="Alice")
    p2 = SimpleNamespace(name="Bob")

    io_mod.saveDataStructure(
        patientList=[p1, p2],
        savingPath="/tmp/patients",
        compressedBool=False,
        splitPatientsBool=True
    )

    assert len(calls) == 2
    assert calls[0][0] == [p1]
    assert calls[1][0] == [p2]
    assert calls[0][1] == "/tmp/patients_Alice"
    assert calls[1][1] == "/tmp/patients_Bob"
    assert calls[0][2] is False
    assert calls[1][2] is False


def test_saveDataStructure_compressed_true(mock_saveSerializedObjects):
    calls = mock_saveSerializedObjects

    p1 = SimpleNamespace(name="Alice")

    io_mod.saveDataStructure(
        patientList=[p1],
        savingPath="/tmp/patients",
        compressedBool=True,
        splitPatientsBool=False
    )

    assert len(calls) == 1
    assert calls[0][2] is True
    
    
def test_saveDataStructure_empty_list_split_true(mock_saveSerializedObjects):
    calls = mock_saveSerializedObjects

    io_mod.saveDataStructure(
        patientList=[],
        savingPath="/tmp/patients",
        compressedBool=False,
        splitPatientsBool=True
    )

    assert len(calls) == 0