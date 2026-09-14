from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pydicom
import pytest

from opentps.core.io.dicomIO import readDicomPET


def _write_minimal_pet_slice(
    path: Path,
    *,
    pixel_array: np.ndarray,
    z_mm: float,
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_mm: tuple[float, float, float] = (2.0, 2.0, 3.0),
    rescale: tuple[float, float] | None = None,
    series_uid: str | None = None,
    study_uid: str | None = None,
    frame_uid: str | None = None,
    series_description: str | None = "TEST_PET",
    patient_id: str = "P001",
    patient_name: str = "Test^Patient",
    patient_sex: str = "O",
    instance_number: int = 1,
) -> Path:
    """
    Write a minimal PET DICOM slice on disk for unit tests.

    Parameters
    ----------
    path: Path
        Output path of the DICOM file.
    pixel_array: np.ndarray
        2D array containing the raw pixel values for one slice.
    z_mm: float
        Slice position (ImagePositionPatient[2]) in mm.
    origin_mm: tuple[float, float, float]
        Origin (x, y, z) in mm. Only x and y are used for ImagePositionPatient.
    spacing_mm: tuple[float, float, float]
        Spacing (x, y, z) in mm. PixelSpacing uses (y, x) order (row, col).
    rescale: tuple[float, float] or None
        (RescaleSlope, RescaleIntercept). If None, tags are omitted.
    series_uid, study_uid, frame_uid: str or None
        UIDs; generated if None.
    series_description: str or None
        SeriesDescription value. If None, tag omitted.
    patient_id, patient_name, patient_sex: str
        Patient tags.
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
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"  # PET Image Storage
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

    ds.Modality = "PT"
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

    if series_description is not None:
        ds.SeriesDescription = series_description

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


def test_readDicomPET_single_slice_no_rescale_uses_raw_pixels(tmp_path: Path, caplog):
    """
    Check that readDicomPET loads raw pixels when Rescale tags are not detected.

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.
    caplog:
        Pytest fixture capturing logs.

    Returns
    -------
    None
    """
    pixels = np.array([[1, 2], [3, 4]], dtype=np.int16)

    dcm_path = _write_minimal_pet_slice(
        tmp_path / "pt_0001.dcm",
        pixel_array=pixels,
        z_mm=0.0,
        rescale=None,
    )

    img = readDicomPET([str(dcm_path)])

    assert img is not None
    assert hasattr(img, "imageArray")
    assert img.imageArray.shape == (pixels.shape[1], pixels.shape[0], 1)

    expected = pixels.astype(np.float32).T
    np.testing.assert_allclose(img.imageArray[:, :, 0], expected, rtol=0, atol=0)

    # Le code log un warning "no RescaleSlope..."
    assert any("no RescaleSlope" in rec.message for rec in caplog.records)


def test_readDicomPET_slices_are_sorted_by_z(tmp_path: Path):
    """
    Check that readDicomPET sorts slices using ImagePositionPatient[2].

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    p0 = np.array([[10, 10], [10, 10]], dtype=np.int16)
    p1 = np.array([[20, 20], [20, 20]], dtype=np.int16)

    # On écrit volontairement dans l'ordre inverse (z=5 puis z=0)
    dcm_z5 = _write_minimal_pet_slice(tmp_path / "pt_z5.dcm", pixel_array=p1, z_mm=5.0, instance_number=2)
    dcm_z0 = _write_minimal_pet_slice(tmp_path / "pt_z0.dcm", pixel_array=p0, z_mm=0.0, instance_number=1)

    img = readDicomPET([str(dcm_z5), str(dcm_z0)])

    assert img is not None
    assert img.imageArray.shape == (2, 2, 2)

    # slice 0 doit correspondre à z=0 → valeurs 10
    np.testing.assert_allclose(img.imageArray[:, :, 0], p0.astype(np.float32).T, rtol=0, atol=0)
    # slice 1 doit correspondre à z=5 → valeurs 20
    np.testing.assert_allclose(img.imageArray[:, :, 1], p1.astype(np.float32).T, rtol=0, atol=0)

    # sliceLocation doit être trié croissant
    assert np.all(np.diff(img.sliceLocation) >= 0)


def test_readDicomPET_patient_information_is_set(tmp_path: Path):
    """
    Check that readDicomPET fills Patient() info when PatientID exists.

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    dcm_path = _write_minimal_pet_slice(
        tmp_path / "pt_patient.dcm",
        pixel_array=np.array([[1, 2], [3, 4]], dtype=np.int16),
        z_mm=0.0,
        patient_id="ABC123",
        patient_name="Doe^Jane",
        patient_sex="F",
    )

    img = readDicomPET([str(dcm_path)])

    assert img is not None
    assert hasattr(img, "patient")
    assert img.patient.id == "ABC123"
    assert str(img.patient.name) == "Doe^Jane"
    assert img.patient.sex == "F"


def test_readDicomPET_name_prefers_series_description(tmp_path: Path):
    """
    Check that image.name uses SeriesDescription when available.

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    dcm_path = _write_minimal_pet_slice(
        tmp_path / "pt_name.dcm",
        pixel_array=np.array([[1, 1], [1, 1]], dtype=np.int16),
        z_mm=0.0,
        series_description="MY_PET_SERIES",
    )

    img = readDicomPET([str(dcm_path)])

    assert img is not None
    assert img.name == "MY_PET_SERIES"


def test_readDicomPET_spacing_uses_pixelspacing_and_mean_slice_distance(tmp_path: Path):
    """
    Check that spacing is (PixelSpacing[1], PixelSpacing[0], meanSliceDistance).

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    sx, sy, sz = 2.0, 3.0, 4.0  # x, y, z

    d0 = _write_minimal_pet_slice(
        tmp_path / "pt_0.dcm",
        pixel_array=np.array([[1, 2], [3, 4]], dtype=np.int16),
        z_mm=0.0,
        spacing_mm=(sx, sy, sz),
        instance_number=1,
    )
    d1 = _write_minimal_pet_slice(
        tmp_path / "pt_1.dcm",
        pixel_array=np.array([[5, 6], [7, 8]], dtype=np.int16),
        z_mm=sz,
        spacing_mm=(sx, sy, sz),
        instance_number=2,
    )

    img = readDicomPET([str(d0), str(d1)])

    assert img is not None
    # pixelSpacing = (PixelSpacing[1], PixelSpacing[0], meanSliceDistance)
    # PixelSpacing dans le fichier = [sy, sx]
    # => attendu = (sx, sy, sz)
    assert img.spacing[0] == pytest.approx(sx)
    assert img.spacing[1] == pytest.approx(sy)
    assert img.spacing[2] == pytest.approx(sz)


@pytest.mark.xfail(reason="Bug in readDicomPET: checks hasattr on a string path instead of reading the first DICOM dataset.")
def test_readDicomPET_rescale_should_be_applied_when_tags_exist(tmp_path: Path):
    """
    Expected behavior (but currently broken): apply RescaleSlope/Intercept when present.

    Parameters
    ----------
    tmp_path: Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    pixels = np.array([[1, 2], [3, 4]], dtype=np.int16)
    slope, intercept = 2.0, 10.0

    dcm_path = _write_minimal_pet_slice(
        tmp_path / "pt_rescale.dcm",
        pixel_array=pixels,
        z_mm=0.0,
        rescale=(slope, intercept),
    )

    img = readDicomPET([str(dcm_path)])

    expected = (pixels.astype(np.float32) * slope + intercept).T
    np.testing.assert_allclose(img.imageArray[:, :, 0], expected, rtol=0, atol=0)