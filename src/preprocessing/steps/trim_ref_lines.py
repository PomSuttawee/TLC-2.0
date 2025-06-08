import cv2
import numpy as np
from typing import Optional, Tuple

_BORDER_MARGIN_PX = 20
_MIN_LINE_LEN_RATIO = 0.25          # 25 % of width
_HEADER_FRAC = 0.15
_FOOTER_FRAC = 0.15                 # prefer symmetric names

def trim_reference_lines(
    image: np.ndarray,
    header_frac: float = _HEADER_FRAC,
    footer_frac: float = _FOOTER_FRAC,
    visualize: bool = False,
) -> np.ndarray:
    """
    Trim a TLC image to the region between the origin line and the solvent front.
    """
    # Trim
    header = image[: int(image.shape[0] * header_frac)]
    footer = image[int(image.shape[0] * (1 - footer_frac)) :]

    header_y = _detect_horizontal_line(header, visualize)
    footer_rel = _detect_horizontal_line(footer, visualize)
    if header_y is None or footer_rel is None:
        raise ValueError("Unable to locate origin or front line.")

    footer_y = footer_rel + image.shape[0] - footer.shape[0]

    return image[header_y:footer_y]

# ---------- helpers ---------- #

_lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

def _detect_horizontal_line(img: np.ndarray, visualize: bool = False) -> Optional[int]:
    lines = _detect_all_lines(img)
    lines = _filter_horizontal_lines(lines, img.shape)
    if not len(lines):
        return None

    if visualize:
        _show_lines(img, lines, title="horizontal lines")

    ys = [line[0][1] for line in lines]
    return int(np.mean(ys))

def _detect_all_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enh = clahe.apply(gray)
    return _lsd.detect(enh)[0]

def _filter_horizontal_lines(lines: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = shape[:2]
    out = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if min(y1, y2) < _BORDER_MARGIN_PX or max(y1, y2) > h - _BORDER_MARGIN_PX:
            continue
        angle = abs(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
        if 5 < angle < 175:
            continue
        if np.hypot(x2 - x1, y2 - y1) < _MIN_LINE_LEN_RATIO * w:
            continue
        out.append([[x1, y1, x2, y2]])
    return np.asarray(out)

def _show_lines(img: np.ndarray, lines: np.ndarray, title: str = "") -> None:
    import matplotlib.pyplot as plt
    vis = img.copy()
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()