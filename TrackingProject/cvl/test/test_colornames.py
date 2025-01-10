from cvl.features import colornames_image

import numpy as np

def test_colornames():
    red_image = np.zeros((128, 128, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255

    c_im = colornames_image(red_image, 'probability')

    assert c_im.shape == (128,128,11)