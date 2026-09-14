import re
import numpy as np
import pydicom
import pytest

from opentps.core.io.dicomIO import writeDicomCT



def _list_dcms(folder):
    """
    Return all DICOM (.dcm) files in a folder.

    Parameters
    ----------
    folder: pathlib.Path
        Output directory containing DICOM files.

    Returns
    -------
    files: list
        Sorted list of DICOM file paths.
    """
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() == ".dcm"),
        key=lambda p: p.name,
    )


def test_writeDicomCT_writes_one_file_per_slice_and_returns_series_uid(tmp_path, ct_factory):
    """
    Check that writeDicomCT writes one DICOM file per slice
    and returns the SeriesInstanceUID.
    """
    arr = np.zeros((2, 2, 3), dtype=np.float32)
    ct = ct_factory(array3d=arr)

    out_dir = tmp_path / "out_ct"
    assert not out_dir.exists()

    returned_uid = writeDicomCT(ct, str(out_dir))

    files = _list_dcms(out_dir)
    assert len(files) == 3
    assert returned_uid == ct.seriesInstanceUID


def test_writeDicomCT_uses_outputFileName_and_sanitizes(tmp_path, ct_factory):
    """
    Check that outputFileName is used for filenames
    and sanitized to alphanumeric characters.
    """
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    ct = ct_factory(array3d=arr)

    out_dir = tmp_path / "out_ct"
    output_name = "CT Name: (weird)/\\ stuff!! 123"

    writeDicomCT(ct, str(out_dir), outputFileName=output_name)

    files = _list_dcms(out_dir)
    assert len(files) == 2

    sanitized = "".join(ch for ch in output_name if ch.isalnum())
    assert all(
        re.match(rf"CT_{re.escape(sanitized)}_\d{{4}}\.dcm$", f.name)
        for f in files
    )


def test_writeDicomCT_sets_slice_specific_tags(tmp_path, ct_factory):
    """
    Check that InstanceNumber, ImagePositionPatient and SliceLocation
    are correctly written according to origin and spacing.
    """
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    ct = ct_factory(
        array3d=arr,
        spacing=(1.0, 1.0, 2.5),
        origin=(0.0, 0.0, 10.0),
    )

    out_dir = tmp_path / "out_ct"
    writeDicomCT(ct, str(out_dir), outputFileName="abc")

    files = _list_dcms(out_dir)
    assert len(files) == 2

    d0 = pydicom.dcmread(str(files[0]))
    d1 = pydicom.dcmread(str(files[1]))

    assert int(d0.InstanceNumber) == 1
    assert int(d1.InstanceNumber) == 2

    assert float(d0.ImagePositionPatient[2]) == pytest.approx(10.0)
    assert float(d1.ImagePositionPatient[2]) == pytest.approx(12.5)

    assert float(d0.SliceLocation) == pytest.approx(10.0)
    assert float(d1.SliceLocation) == pytest.approx(12.5)


def test_writeDicomCT_pixeldata_written_as_int16_with_transpose(tmp_path, ct_factory):
    """
    Check that pixel data is written as int16
    and transposed according to implementation.
    """
    arr = np.zeros((2, 3, 1), dtype=np.float32)
    arr[:, :, 0] = np.array(
        [[1, 2, 3],
         [4, 5, 6]],
        dtype=np.float32,
    )

    ct = ct_factory(array3d=arr)
    out_dir = tmp_path / "out_ct"

    writeDicomCT(ct, str(out_dir), outputFileName="pix")

    files = _list_dcms(out_dir)
    assert len(files) == 1

    dcm = pydicom.dcmread(str(files[0]))
    px = dcm.pixel_array.astype(np.int16)

    expected = np.round(arr[:, :, 0]).T.astype(np.int16)
    assert px.shape == expected.shape
    assert np.array_equal(px, expected)


def test_writeDicomCT_rescales_when_values_outside_int16_and_roundtrip_recovers(tmp_path, ct_factory):
    """
    Check that values outside int16 range trigger rescaling
    and that reconstructed values match the original data.
    """
    original = np.array(
        [[-50000.0, -40000.0],
         [ 40000.0,  50000.0]],
        dtype=np.float32,
    )
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    arr[:, :, 0] = original
    arr[:, :, 1] = original + 123.0

    ct = ct_factory(array3d=arr)
    out_dir = tmp_path / "out_ct"

    writeDicomCT(ct, str(out_dir), outputFileName="rs")

    files = _list_dcms(out_dir)
    assert len(files) == 2

    d0 = pydicom.dcmread(str(files[0]))
    slope = float(d0.RescaleSlope)
    intercept = float(d0.RescaleIntercept)

    assert not (slope == 1.0 and intercept == 0.0)

    stored = d0.pixel_array.astype(np.float32)
    recon = stored * slope + intercept

    assert np.allclose(recon, original.T, atol=1.0)