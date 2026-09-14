import numpy as np
import pytest

from opentps.core.data.images import CTImage


def make_ct_image(
    *,
    array3d: np.ndarray,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 2.5),
    series_uid: str | None = None,
    frame_uid: str | None = None,
    sop_uids=None,
    name: str = "test_ct",
) -> CTImage:
    """
    Create a CTImage object for DICOM unit tests.

    Parameters
    ----------
    array3d: numpy.ndarray
        3D array of shape (X, Y, Z) representing the CT volume.
    origin: tuple
        Image origin in patient coordinates (x, y, z).
    spacing: tuple
        Voxel spacing in mm (x, y, z).
    series_uid: str
        SeriesInstanceUID to assign to the CT image.
    frame_uid: str
        FrameOfReferenceUID to assign to the CT image.
    sop_uids: list
        Optional list of SOPInstanceUIDs.
    name: str
        Optional image name.
    """
    assert array3d.ndim == 3
    z = array3d.shape[2]
    slice_location = np.array(
        [origin[2] + i * spacing[2] for i in range(z)],
        dtype=float,
    )

    return CTImage(
        imageArray=array3d.astype(np.float32),
        origin=np.array(origin, dtype=float),
        spacing=np.array(spacing, dtype=float),
        sliceLocation=slice_location,
        seriesInstanceUID=series_uid or "1.2.3.4.5",
        frameOfReferenceUID=frame_uid or "9.8.7.6.5",
        sopInstanceUIDs=sop_uids,
        name=name,
    )


@pytest.fixture
def ct_image() -> CTImage:
    array = np.zeros((8, 8, 3), dtype=np.int16)
    array[:, :, 1] = 500

    return make_ct_image(
        array3d=array,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 2.5),
        series_uid="1.2.3.4.5.6",
        frame_uid="9.8.7.6.5.4",
    )


@pytest.fixture
def ct_factory():
    def _factory(
        *,
        array3d: np.ndarray,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 2.5),
        series_uid: str | None = None,
        frame_uid: str | None = None,
        sop_uids=None,
        name: str = "test_ct",
    ) -> CTImage:
        return make_ct_image(
            array3d=array3d,
            origin=origin,
            spacing=spacing,
            series_uid=series_uid,
            frame_uid=frame_uid,
            sop_uids=sop_uids,
            name=name,
        )

    return _factory