from .steps.crop_plate import crop_plate
from .steps.trim_ref_lines import trim_reference_lines
from .steps.segment_spots import segment_spots

def full_preprocess(img):
    img = crop_plate(img)
    img = trim_reference_lines(img)
    return segment_spots(img)