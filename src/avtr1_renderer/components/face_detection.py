# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""InsightFace SCRFD face detector wrapper.

Wraps a ``FaceDetEngine`` plus the numpy-side post-processing layer
(blob preprocess + anchor decode + NMS) that returns boxes/keypoints in
the original image coordinates.

The detector is BGR-blob in (mean=127.5, std=128.0, swapRB=True), which
is the InsightFace SCRFD preprocess. We replicate it via
``cv2.dnn.blobFromImage`` to bit-match the reference.

I/O contracts live in ``avtr1_renderer.models.face_detection``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from avtr1_renderer.models.face_detection import FaceDetEngine, FaceDetInput

INPUT_SIZE = (512, 512)
INPUT_MEAN = 127.5
INPUT_STD = 128.0
FEAT_STRIDE_FPN = (8, 16, 32)
NUM_ANCHORS = 2

# Anchor-grid memo. Process-wide -- the keys are (height, width, stride)
# tuples and there are exactly three of them (one per FPN level), so this
# never grows past three entries.
_CENTER_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    """Same NMS as InsightFace reference (with the ``+1`` quirk)."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def _run(blob_chw: np.ndarray, *, det: FaceDetEngine) -> dict[str, np.ndarray]:
    """``blob_chw``: (1, 3, 512, 512) float32. Returns dict of all 9 outputs
    copied to host memory (post-processing is on CPU)."""
    inp = torch.from_numpy(blob_chw).cuda().contiguous()
    out = det(FaceDetInput(image=inp))
    torch.cuda.synchronize()
    return {
        "scores1": out.scores1.cpu().numpy(),
        "scores2": out.scores2.cpu().numpy(),
        "scores3": out.scores3.cpu().numpy(),
        "boxes1": out.boxes1.cpu().numpy(),
        "boxes2": out.boxes2.cpu().numpy(),
        "boxes3": out.boxes3.cpu().numpy(),
        "kps1": out.kps1.cpu().numpy(),
        "kps2": out.kps2.cpu().numpy(),
        "kps3": out.kps3.cpu().numpy(),
    }


def _decode(
    outputs: dict[str, np.ndarray],
    det_thresh: float = 0.5,
    input_height: int = 512,
    input_width: int = 512,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    scores_list: list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    kpss_list: list[np.ndarray] = []

    for idx, stride in enumerate(FEAT_STRIDE_FPN):
        level = idx + 1
        scores = outputs[f"scores{level}"]
        bbox_preds = outputs[f"boxes{level}"] * stride
        kps_preds = outputs[f"kps{level}"] * stride

        height = input_height // stride
        width = input_width // stride
        key = (height, width, stride)
        if key in _CENTER_CACHE:
            anchor_centers = _CENTER_CACHE[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if NUM_ANCHORS > 1:
                anchor_centers = np.stack([anchor_centers] * NUM_ANCHORS, axis=1).reshape(
                    (-1, 2)
                )
            _CENTER_CACHE[key] = anchor_centers

        pos_inds = np.where(scores >= det_thresh)[0]
        bboxes = _distance2bbox(anchor_centers, bbox_preds)
        kpss = _distance2kps(anchor_centers, kps_preds).reshape(kps_preds.shape[0], -1, 2)
        scores_list.append(scores[pos_inds])
        bboxes_list.append(bboxes[pos_inds])
        kpss_list.append(kpss[pos_inds])

    return scores_list, bboxes_list, kpss_list


def detect_faces(
    img_rgb: np.ndarray,
    *,
    det: FaceDetEngine,
    det_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SCRFD face detection on an HWC RGB uint8 image.

    Args:
        img_rgb: HWC uint8 image.
        det: SCRFD engine (e.g. built via
            ``load_engine(path, FaceDetInput, FaceDetOutput)``).
        det_thresh: per-box score threshold.

    Returns:
        ``(det, kpss)`` where ``det`` is ``(N, 5)`` in
        ``(x1, y1, x2, y2, score)`` form, and ``kpss`` is ``(N, 5, 2)``.
        Both are numpy arrays scaled back to the original image
        coordinates.
    """
    h, w = img_rgb.shape[:2]
    in_w, in_h = INPUT_SIZE
    im_ratio = float(h) / w
    model_ratio = float(in_h) / in_w
    if im_ratio > model_ratio:
        new_height = in_h
        new_width = int(new_height / im_ratio)
    else:
        new_width = in_w
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / h
    resized = cv2.resize(img_rgb, (new_width, new_height))
    det_img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized

    blob = cv2.dnn.blobFromImage(
        det_img,
        1.0 / INPUT_STD,
        (in_w, in_h),
        (INPUT_MEAN, INPUT_MEAN, INPUT_MEAN),
        swapRB=True,
    )

    outputs = _run(blob, det=det)
    scores_list, bboxes_list, kpss_list = _decode(
        outputs, det_thresh=det_thresh, input_height=in_h, input_width=in_w
    )

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = _nms(pre_det)
    det_out = pre_det[keep, :]
    kpss = kpss[order, :, :][keep, :, :]
    return det_out, kpss
