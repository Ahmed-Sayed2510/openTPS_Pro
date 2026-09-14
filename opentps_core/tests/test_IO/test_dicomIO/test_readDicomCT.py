import logging
import numpy as np
import pydicom
import pytest

from opentps.core.io.dicomIO import readDicomCT


def _write_ct_slice(
    path,
    *,
    z: float,
    pixel_array: np.ndarray,
    slope: float = 1.0,
    intercept: float = 0.0,
    pixel_spacing=(1.0, 1.0),
    slice_thickness: float = 2.5,
    series_uid: str | None = None,
    frame_uid: str | None = None,
    series_description: str | None = None,
    patient_id: str | None = "P123",
):
    """
    Create a minimal *working* CT DICOM slice for readDicomCT tests.

    Important: readDicomCT accesses some tags WITHOUT hasattr checks:
      - ImagePositionPatient
      - PixelSpacing
      - RescaleSlope / RescaleIntercept
      - SOPInstanceUID
      - SeriesInstanceUID
      - Rows / Columns
      - pixel_array (requires PixelData + transfer syntax + basic pixel tags)

    So we always set those.
    """
    if series_uid is None:
        series_uid = pydicom.uid.generate_uid()
    if frame_uid is None:
        frame_uid = pydicom.uid.generate_uid()

    # ---- File meta (enough to allow pydicom to decode PixelData) ----
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = pydicom.dataset.FileDataset(
        str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )

    # ---- Required identity tags ----
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = frame_uid

    # ---- Patient (only used if PatientID exists) ----
    if patient_id is not None:
        ds.PatientID = patient_id
        ds.PatientName = "Test^Patient"
        ds.PatientBirthDate = "19900101"
        ds.PatientSex = "O"

    # ---- Geometry / spacing (readDicomCT uses PixelSpacing + ImagePositionPatient) ----
    rows, cols = pixel_array.shape
    ds.Rows = int(rows)
    ds.Columns = int(cols)

    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.SliceThickness = float(slice_thickness)

    # ---- Intensity conversion (readDicomCT uses these without hasattr) ----
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)

    # ---- Minimal pixel module so ds.pixel_array works ----
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16

    ds.Modality = "CT"
    ds.InstanceNumber = 1
    ds.SeriesNumber = 1

    # pydicom needs these flags consistent with TransferSyntax
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PixelData = pixel_array.astype(np.int16).tobytes()

    if series_description is not None:
        ds.SeriesDescription = series_description

    ds.save_as(str(path))
    return ds.SOPInstanceUID


@pytest.fixture
def ct_series(tmp_path):
    """
    Create a minimal CT DICOM series for readDicomCT tests.

    Slices are intentionally written out-of-order (z=30,10,20)
    so we can verify sorting by slice location.
    """
    px = np.zeros((2, 2), dtype=np.int16)

    p30 = tmp_path / "slice30.dcm"
    p10 = tmp_path / "slice10.dcm"
    p20 = tmp_path / "slice20.dcm"

    # force same series/frame so it's coherent (optional but clean)
    series_uid = pydicom.uid.generate_uid()
    frame_uid = pydicom.uid.generate_uid()

    uid30 = _write_ct_slice(p30, z=30.0, pixel_array=px, series_uid=series_uid, frame_uid=frame_uid)
    uid10 = _write_ct_slice(p10, z=10.0, pixel_array=px, series_uid=series_uid, frame_uid=frame_uid)
    uid20 = _write_ct_slice(p20, z=20.0, pixel_array=px, series_uid=series_uid, frame_uid=frame_uid)

    yield {
        "paths": [str(p30), str(p10), str(p20)],
        "sorted_z": [10.0, 20.0, 30.0],
        "sorted_uids": [uid10, uid20, uid30],
        "series_uid": series_uid,
        "frame_uid": frame_uid,
    }


def test_readDicomCT_sorts_slices(ct_series):
    image = readDicomCT(ct_series["paths"])

    assert list(image.sliceLocation) == ct_series["sorted_z"]
    assert image.sopInstanceUIDs == ct_series["sorted_uids"]


def test_readDicomCT_applies_rescale(tmp_path):
    px = (np.ones((2, 2), dtype=np.int16) * 10)
    slope = 2.0
    intercept = -100.0

    p0 = tmp_path / "z0.dcm"
    p5 = tmp_path / "z5.dcm"

    _write_ct_slice(p0, z=0.0, pixel_array=px, slope=slope, intercept=intercept)
    _write_ct_slice(p5, z=5.0, pixel_array=px, slope=slope, intercept=intercept)

    image = readDicomCT([str(p0), str(p5)])

    expected = px.astype(np.float32) * slope + intercept

    assert image.imageArray.shape == (2, 2, 2)
    assert np.allclose(image.imageArray[:, :, 0], expected)
    assert np.allclose(image.imageArray[:, :, 1], expected)


def test_readDicomCT_spacing(tmp_path):
    px = np.zeros((2, 2), dtype=np.int16)

    p0 = tmp_path / "z0.dcm"
    p10 = tmp_path / "z10.dcm"

    _write_ct_slice(p0, z=0.0, pixel_array=px)
    _write_ct_slice(p10, z=10.0, pixel_array=px)

    image = readDicomCT([str(p0), str(p10)])

    assert np.isclose(image.spacing[2], 10.0)


def test_readDicomCT_name_from_series_description(tmp_path):
    px = np.zeros((2, 2), dtype=np.int16)

    p0 = tmp_path / "z0.dcm"
    p5 = tmp_path / "z5.dcm"

    _write_ct_slice(p0, z=0.0, pixel_array=px, series_description="MySeries")
    _write_ct_slice(p5, z=5.0, pixel_array=px, series_description="MySeries")

    image = readDicomCT([str(p0), str(p5)])

    assert image.name == "MySeries"


def test_readDicomCT_fallback_series_uid(tmp_path):
    px = np.zeros((2, 2), dtype=np.int16)
    series_uid = pydicom.uid.generate_uid()

    p0 = tmp_path / "z0.dcm"
    p5 = tmp_path / "z5.dcm"

    _write_ct_slice(p0, z=0.0, pixel_array=px, series_uid=series_uid)
    _write_ct_slice(p5, z=5.0, pixel_array=px, series_uid=series_uid)

    image = readDicomCT([str(p0), str(p5)])

    assert image.name == series_uid


def test_readDicomCT_warns_on_slice_thickness_mismatch(tmp_path, caplog):
    px = np.zeros((2, 2), dtype=np.int16)

    p0 = tmp_path / "z0.dcm"
    p10 = tmp_path / "z10.dcm"

    _write_ct_slice(p0, z=0.0, pixel_array=px, slice_thickness=2.5)
    _write_ct_slice(p10, z=10.0, pixel_array=px, slice_thickness=2.5)

    dicom_logger = logging.getLogger(readDicomCT.__module__)
    dicom_logger.addHandler(caplog.handler)
    dicom_logger.setLevel(logging.WARNING)

    readDicomCT([str(p0), str(p10)])

    dicom_logger.removeHandler(caplog.handler)

    assert any("Mean Slice Distance" in r.message for r in caplog.records)