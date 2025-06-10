import cv2
import numpy as np

def segment_spots(trimmed_bgr: np.ndarray):
    norm       = cv2.divide(trimmed_bgr,
                            cv2.GaussianBlur(trimmed_bgr,(0,0),35), scale=255)
    lab        = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    _, a, b    = cv2.split(lab)
    chroma     = np.sqrt((a.astype(np.int16)-128)**2 +
                         (b.astype(np.int16)-128)**2).astype(np.uint8)

    T, _       = cv2.threshold(chroma, 0, 255, cv2.THRESH_OTSU)
    hi         = (chroma >= T).astype(np.uint8)*255
    lo         = (chroma >= 0.5*T).astype(np.uint8)*255
    mask       = cv2.morphologyEx(hi | lo, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask       = cv2.medianBlur(mask, 5)

    cnts, _    = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts       = [c for c in cnts if 30 < cv2.contourArea(c) < 0.2*mask.size]

    spot_mask  = np.zeros_like(mask)
    cv2.drawContours(spot_mask, cnts, -1, 255, -1)

    return spot_mask, cnts        # mask for pixel-wise ops, cnts for geometry

def separate_overlaps(mask: np.ndarray,
                      area_hi: int = 3000,
                      width_hi: int = 60,
                      peak_ratio: float = 0.45) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Split merged TLC blobs in an existing binary mask (255 = spot, 0 = background).

    Returns
    -------
    clean_mask : np.ndarray
        Same size as `mask`, but with fused blobs separated.
    contours   : list[np.ndarray]
        Contours of every *individual* spot after the split (for geometry/Rf).
    """
    clean_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    next_label = 1                                                       # for visual sanity check
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y:y+h, x:x+w]

        # --- decide whether this blob *could* be multiple spots ---
        if cv2.contourArea(cnt) < area_hi and w < width_hi:
            # small enough – keep as is
            clean_mask[y:y+h, x:x+w][roi > 0] = 255
            continue

        # --- 1. distance transform inside the ROI ---
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

        # --- 2. find peaks that stand out ---
        peak_thresh = dist.max() * peak_ratio
        peaks = (dist > peak_thresh).astype(np.uint8)
        n_seeds, markers = cv2.connectedComponents(peaks)

        if n_seeds <= 2:
            # not enough evidence → leave untouched
            clean_mask[y:y+h, x:x+w][roi > 0] = 255
            continue

        # --- 3. prep markers for watershed ---
        markers = markers + 1                      # background=1 instead of 0
        markers[roi == 0] = 0                      # enforce background outside blob

        # watershed needs 3-channel input, but only geometry matters
        wshed = cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)

        # --- 4. write every region back as an individual spot ---
        for lbl in range(2, n_seeds + 1):
            clean_mask[y:y+h, x:x+w][wshed == lbl] = 255
            next_label += 1    # (optional) could paint different colours for debugging

    # --- 5. re-extract contours of the separated spots ---
    final_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return clean_mask, final_cnts
