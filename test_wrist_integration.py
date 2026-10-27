import unittest,xml.etree.ElementTree as E
import numpy as np
import opensim as o
from wrist_muscle_integration import OUT,BASE,NAMES

class WristIntegrationTests(unittest.TestCase):
    def test_source_wrist_capacity_not_inflated(self):
        a=E.parse(OUT/'mobl_source.osim');b=E.parse(OUT/'scaled_source.osim')
        for name in NAMES:
            x=a.find(f".//ForceSet/objects/*[@name='{name}']/max_isometric_force")
            y=b.find(f".//ForceSet/objects/*[@name='{name}']/max_isometric_force")
            self.assertEqual(float(x.text),float(y.text))

    def test_existing_capacity_and_inertia_unchanged(self):
        a=E.parse(BASE/'scaled_source.osim');b=E.parse(OUT/'scaled_source.osim')
        for body in a.findall('.//BodySet/objects/*'):
            other=b.find(f".//BodySet/objects/*[@name='{body.get('name')}']")
            for field in ['mass','mass_center','inertia']:self.assertEqual(body.findtext(field),other.findtext(field))
        for mus in a.findall('.//ForceSet/objects/*'):
            other=b.find(f".//ForceSet/objects/*[@name='{mus.get('name')}']")
            for field in ['max_isometric_force','optimal_fiber_length','tendon_slack_length']:self.assertEqual(mus.findtext(field),other.findtext(field))

    def test_wrist_muscles_are_connected_and_differentiable(self):
        o.Logger.setLevelString('error');m=o.Model(str(OUT/'scaled_source.osim'));s=m.initSystem()
        self.assertEqual(m.getMuscles().getSize(),39)
        self.assertEqual(m.getNumCoordinates(),21)
        c=m.getCoordinateSet().get('flexion');c.setValue(s,.1,False);m.realizePosition(s)
        lengths=[]
        for v in [.1-1e-6,.1+1e-6]:
            c.setValue(s,v,False);m.realizePosition(s)
            lengths.append([m.getMuscles().get(n).getLength(s) for n in NAMES])
        deriv=(np.array(lengths[1])-lengths[0])/2e-6
        c.setValue(s,.1,False);m.realizePosition(s)
        moment=np.array([m.getMuscles().get(n).computeMomentArm(s,c) for n in NAMES])
        self.assertTrue(np.isfinite(deriv).all())
        self.assertGreater(np.max(abs(deriv)),.001)
        # Coupler constraints alter generalized moment arm; path derivatives
        # above intentionally hold dependent coordinates fixed. Check that
        # the physical path responds, without asserting these are identical.
        self.assertTrue(np.isfinite(moment).all())

if __name__=='__main__':unittest.main()
