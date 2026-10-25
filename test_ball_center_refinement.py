import unittest
import numpy as np
from ball_center_refinement import center

class CenterRefinementTests(unittest.TestCase):
 def scene(self,position=(50.3,49.7)):
  y,x=np.indices((110,110));bg=20+.05*x+60*(abs(x-y)<2);signal=150*np.exp(-.5*(((x-position[0])/7)**2+((y-position[1])/4)**2));return bg+signal,bg
 def test_background_line_removed(self):
  a,b=self.scene()
  for mode in ('binary','weighted','gaussian'):
   r=center(a,b,(50,50),20,mode)
   self.assertEqual(r['status'],'candidate');self.assertLess(np.linalg.norm(np.array(r['pixel'])-[50.3,49.7]),.2)
 def test_gaussian_recovers_subpixel(self):
  a,b=self.scene((52.38,47.61));r=center(a,b,(50,50),20,'gaussian');np.testing.assert_allclose(r['pixel'],[52.38,47.61],atol=.001)
 def test_missing_and_boundary_preserved(self):
  self.assertEqual(center(np.zeros((100,100)),np.zeros((100,100)),(50,50))['status'],'no_component')
  a,b=self.scene();self.assertEqual(center(a,b,(5,5))['status'],'image_boundary')
 def test_invalid_shape(self):
  with self.assertRaises(ValueError):center(np.zeros((3,4)),np.zeros((4,3)),(2,2))
 def test_invalid_parameters(self):
  a,b=self.scene()
  for kwargs in ({'threshold':float('nan')},{'radius':24.5},{'threshold':float('inf')}):
   with self.assertRaises(ValueError):center(a,b,(50,50),**kwargs)
 def test_nearest_component(self):
  a,b=self.scene();a[70:80,70:80]+=220;r=center(a,b,(50,50));self.assertLess(np.linalg.norm(np.array(r['pixel'])-[50.3,49.7]),.2)
if __name__=='__main__':unittest.main()
