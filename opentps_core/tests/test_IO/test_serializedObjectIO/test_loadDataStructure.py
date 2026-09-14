import pytest
import opentps.core.io.serializedObjectIO as sio

def test_loadDataStructure_invalid_pickle_content_raises(tmp_path, monkeypatch):
    """
    Ensure that loadDataStructure raises an exception when the file contains invalid pickle data.

    The function tries pickle.loads first, then falls back to opentps.core.utils.pickel2.loads.
    This test forces both loaders to fail to validate that an exception is propagated.
    """

    # Create a temporary file with invalid pickle content
    invalid_file = tmp_path / "invalid.pkl"
    invalid_file.write_bytes(b"not a valid pickle content")

    # Force pickle.loads to raise an exception
    def _raise_pickle_error(*_args, **_kwargs):
        raise Exception("pickle failed")

    monkeypatch.setattr(sio.pickle, "loads", _raise_pickle_error)

    # Inject a fake pickel2 module that also raises
    class FakePickel2Module:
        @staticmethod
        def loads(_bytes):
            raise Exception("pickle2 failed")

    monkeypatch.setitem(
        __import__("sys").modules,
        "opentps.core.utils.pickel2",
        FakePickel2Module,
    )

    # Verify that an exception is raised
    with pytest.raises(Exception):
        sio.loadDataStructure(str(invalid_file))
