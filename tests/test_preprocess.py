import cv2
from test_utils import get_test_image
from tlc_analyzer.preprocess import outer_structure, inner_structure

images = get_test_image()

for name, image in images.items():
    image = cv2.imread(name)
    processed_image = outer_structure.process_tlc_paper(image)
    cv2.imshow(f"Processed Image: {name}", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    processed_image = inner_structure.process_tlc_paper(processed_image)