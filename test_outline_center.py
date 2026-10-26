import unittest
import numpy as np
from PIL import Image
from outline_center_test import outline

class OutlineTest(unittest.TestCase):
    def test_known_ellipse(self):
        yy,xx=np.indices((80,100));mask=((xx-50)/10)**2+((yy-40)/6)**2<=1
        arr=np.full((80,100,3),30,np.uint8);arr[mask]=200
        result=outline(Image.fromarray(arr),(50,40),.5)
        self.assertEqual(result['status'],'candidate');np.testing.assert_allclose(result['pixel'],[50,40],atol=.1)
    def test_background_and_boundary_rejected(self):
        im=Image.new('RGB',(100,80),(20,20,20));self.assertEqual(outline(im,(50,40),.5)['status'],'low_contrast')
        a=np.full((80,100,3),20,np.uint8);a[36:45,20:80]=200
        self.assertEqual(outline(Image.fromarray(a),(50,40),.5)['status'],'roi_boundary')
    def test_invalid(self):
        with self.assertRaises(ValueError):outline(Image.new('RGB',(100,80)),(50,40),1)

if __name__=='__main__':unittest.main()
