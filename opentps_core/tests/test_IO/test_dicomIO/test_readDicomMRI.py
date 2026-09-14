from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pydicom
import pytest

from opentps.core.io.dicomIO import readDicomMRI


def _write_minimal_mr_slice(
    path: Path,
    *,
    pixel_array: np.ndarray,
    z_mm: float,
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 2.5),
    rescale: tuple[float, float] | None = (1.0, 0.0),
    series_uid: str | None = None,
    study_uid: str | None = None,
    frame_uid: str | None = None,
    patient_id: str = "P001",
    patient_name: str = "Test^Patient",
    patient_sex: str = "O",
    instance_number: int = 1,
) -> Path:
    """
    Write a minimal MR DICOM slice on disk for unit tests.

    Parameters
    ----------
    path: Path
        Output path of the DICOM file.
    pixel_array: np.ndarray
        2D array containing the raw pixel values for one slice.
    z_mm: float
        Slice position (ImagePositionPatient[2]) in mm.
    origin_mm: tuple
        Origin (x, y, z) in mm. Only x and y are used for ImagePositionPatient.
    spacing_mm: tuple
        Spacing (x, y, z) in mm. PixelSpacing uses (y, x) order (row, col).
    rescale: tuple or None
        (RescaleSlope, RescaleIntercept) to write. If None, rescale tags are omitted.
    series_uid: str, optional
        SeriesInstanceUID. Generated if None.
    study_uid: str, optional
        StudyInstanceUID. Generated if None.
    frame_uid: str, optional
        FrameOfReferenceUID. Generated if None.
    patient_id: str
        PatientID value.
    patient_name: str
        PatientName value.
    patient_sex: str
        PatientSex value.
    instance_number: int
        InstanceNumber value.

    Returns
    -------
    path: Path
        Path of the written file.
    """
    path = Path(path)

    arr = np.asarray(pixel_array)
    if arr.ndim != 2:
        raise ValueError("pixel_array must be a 2D array (one slice).")

    arr = arr.astype(np.int16, copy=False)
    rows, cols = arr.shape

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    file_meta.ImplementationVersionName = "OpenTPS_TEST"

    ds = pydicom.dataset.FileDataset(
        filename_or_obj=str(path),
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.StudyInstanceUID = study_uid or pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = series_uid or pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = frame_uid or pydicom.uid.generate_uid()

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    ds.Modality = "MR"
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientSex = patient_sex

    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S.%f")
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S.%f")
    ds.SeriesNumber = 1
    ds.InstanceNumber = int(instance_number)

    ox, oy, _oz = origin_mm
    sx, sy, sz = spacing_mm

    ds.ImagePositionPatient = [float(ox), float(oy), float(z_mm)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [float(sy), float(sx)]
    ds.SliceThickness = float(sz)
    ds.SpacingBetweenSlices = float(sz)

    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = arr.tobytes()

    if rescale is not None:
        slope, intercept = rescale
        ds.RescaleSlope = float(slope)
        ds.RescaleIntercept = float(intercept)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)

    return path


def test_readDicomMRI_single_slice_rescale_applied(tmp_path: Path):
    """
    Check that readDicomMRI applies RescaleSlope and RescaleIntercept.
    """
    pixels = np.array([[1, 2], [3, 4]], dtype=np.int16)
    slope = 2.0
    intercept = 10.0

    dcm_path = _write_minimal_mr_slice(
        tmp_path / "mr_0001.dcm",
        pixel_array=pixels,
        z_mm=0.0,
        rescale=(slope, intercept),
    )

    img = readDicomMRI([str(dcm_path)])

    assert img is not None
    assert hasattr(img, "imageArray")
    assert img.imageArray.shape == (pixels.shape[1], pixels.shape[0], 1)

    expected = (pixels.astype(np.float32) * slope + intercept).T
    np.testing.assert_allclose(img.imageArray[:, :, 0], expected, rtol=0, atol=0)


def test_readDicomMRI_multiple_slices_sorted_by_z(tmp_path: Path):
    """
    Check that readDicomMRI sorts slices by ImagePositionPatient[2] (z) before stacking.
    """
    s0 = np.full((2, 3), 10, dtype=np.int16)
    s1 = np.full((2, 3), 20, dtype=np.int16)
    s2 = np.full((2, 3), 30, dtype=np.int16)

    spacing_z = 2.5

    p2 = _write_minimal_mr_slice(
        tmp_path / "mr_z5.dcm",
        pixel_array=s2,
        z_mm=5.0,
        spacing_mm=(1.0, 1.0, spacing_z),
        rescale=(1.0, 0.0),
        instance_number=3,
    )
    p0 = _write_minimal_mr_slice(
        tmp_path / "mr_z0.dcm",
        pixel_array=s0,
        z_mm=0.0,
        spacing_mm=(1.0, 1.0, spacing_z),
        rescale=(1.0, 0.0),
        instance_number=1,
    )
    p1 = _write_minimal_mr_slice(
        tmp_path / "mr_z25.dcm",
        pixel_array=s1,
        z_mm=2.5,
        spacing_mm=(1.0, 1.0, spacing_z),
        rescale=(1.0, 0.0),
        instance_number=2,
    )

    img = readDicomMRI([str(p2), str(p0), str(p1)])

    assert img is not None
    assert hasattr(img, "imageArray")
    assert img.imageArray.shape == (s0.shape[1], s0.shape[0], 3)

    np.testing.assert_allclose(img.imageArray[:, :, 0], s0.T, rtol=0, atol=0)
    np.testing.assert_allclose(img.imageArray[:, :, 1], s1.T, rtol=0, atol=0)
    np.testing.assert_allclose(img.imageArray[:, :, 2], s2.T, rtol=0, atol=0)

    assert hasattr(img, "sliceLocation")
    np.testing.assert_allclose(
        img.sliceLocation,
        np.array([0.0, 2.5, 5.0], dtype=float),
        rtol=0,
        atol=1e-6,
    )


def test_readDicomMRI_without_rescale_uses_raw_pixels_and_warns(tmp_path: Path, caplog):
    """
    Check that readDicomMRI falls back to raw pixel data when RescaleSlope is missing
    and logs a warning.
    """
    s0 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    s1 = np.array([[5, 6], [7, 8]], dtype=np.int16)

    p0 = _write_minimal_mr_slice(
        tmp_path / "mr_norescale_0.dcm",
        pixel_array=s0,
        z_mm=0.0,
        rescale=None,
        instance_number=1,
    )
    p1 = _write_minimal_mr_slice(
        tmp_path / "mr_norescale_1.dcm",
        pixel_array=s1,
        z_mm=2.5,
        rescale=None,
        instance_number=2,
    )

    with caplog.at_level("WARNING"):
        img = readDicomMRI([str(p0), str(p1)])

    assert img is not None
    assert hasattr(img, "imageArray")
    assert img.imageArray.shape == (s0.shape[1], s0.shape[0], 2)

    np.testing.assert_allclose(img.imageArray[:, :, 0], s0.T.astype(np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(img.imageArray[:, :, 1], s1.T.astype(np.float32), rtol=0, atol=0)

    assert "no RescaleSlope" in caplog.text


def test_readDicomMRI_uses_mean_slice_distance_for_z_spacing(tmp_path: Path):
    """
    Check that readDicomMRI computes Z spacing from the mean distance between slice locations.
    """
    px = np.zeros((2, 2), dtype=np.int16)

    p0 = _write_minimal_mr_slice(
        tmp_path / "mr_z0.dcm",
        pixel_array=px,
        z_mm=0.0,
        rescale=(1.0, 0.0),
        instance_number=1,
        spacing_mm=(1.0, 1.0, 2.5),
    )
    p2 = _write_minimal_mr_slice(
        tmp_path / "mr_z2.dcm",
        pixel_array=px,
        z_mm=2.0,
        rescale=(1.0, 0.0),
        instance_number=2,
        spacing_mm=(1.0, 1.0, 2.5),
    )
    p10 = _write_minimal_mr_slice(
        tmp_path / "mr_z10.dcm",
        pixel_array=px,
        z_mm=10.0,
        rescale=(1.0, 0.0),
        instance_number=3,
        spacing_mm=(1.0, 1.0, 2.5),
    )

    img = readDicomMRI([str(p0), str(p2), str(p10)])

    assert img is not None
    assert hasattr(img, "spacing")
    assert np.isclose(img.spacing[2], 5.0)


def test_readDicomMRI_uses_series_description_as_name_when_present(tmp_path: Path):
    """
    Check that readDicomMRI uses SeriesDescription as image name when present and non-empty.
    """
    px = np.zeros((2, 2), dtype=np.int16)

    series_uid = pydicom.uid.generate_uid()
    study_uid = pydicom.uid.generate_uid()
    frame_uid = pydicom.uid.generate_uid()

    p0 = _write_minimal_mr_slice(
        tmp_path / "mr_desc_z0.dcm",
        pixel_array=px,
        z_mm=0.0,
        rescale=(1.0, 0.0),
        series_uid=series_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        instance_number=1,
    )

    ds = pydicom.dcmread(str(p0))
    ds.SeriesDescription = "My MR Series"
    ds.save_as(str(p0), write_like_original=False)

    img = readDicomMRI([str(p0)])

    assert img is not None
    assert hasattr(img, "name")
    assert img.name == "My MR Series"


def test_readDicomMRI_falls_back_to_series_instance_uid_when_no_description(tmp_path: Path):
    """
    Check that readDicomMRI uses SeriesInstanceUID as image name
    when SeriesDescription is missing or empty.
    """
    px = np.zeros((2, 2), dtype=np.int16)

    series_uid = pydicom.uid.generate_uid()
    study_uid = pydicom.uid.generate_uid()
    frame_uid = pydicom.uid.generate_uid()

    p0 = _write_minimal_mr_slice(
        tmp_path / "mr_uid_fallback.dcm",
        pixel_array=px,
        z_mm=0.0,
        rescale=(1.0, 0.0),
        series_uid=series_uid,
        study_uid=study_uid,
        frame_uid=frame_uid,
        instance_number=1,
    )

    ds = pydicom.dcmread(str(p0))
    if hasattr(ds, "SeriesDescription"):
        del ds.SeriesDescription
    ds.save_as(str(p0), write_like_original=False)

    img = readDicomMRI([str(p0)])

    assert img is not None
    assert hasattr(img, "name")
    assert img.name == series_uid