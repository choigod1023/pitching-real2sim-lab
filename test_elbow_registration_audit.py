import json,unittest,xml.etree.ElementTree as E
import numpy as np
import opensim as o
from elbow_registration_audit import INPUT,OUT,WRIST,ELBOW,elbow_anchor,axes

class RegistrationTests(unittest.TestCase):
    def test_scaled_existing_joint_axes_match_source(self):
        original=E.parse(INPUT.parent/'anatomical-shoulder/TSM_subject_noWeight.osim');scaled=E.parse(INPUT/'scaled_source.osim')
        def axes_by_name(tree,joint):
            j=tree.find(f".//JointSet/objects/*[@name='{joint}']")
            return {x.findtext('coordinates'):np.fromstring(x.findtext('axis'),sep=' ') for x in j.findall('.//TransformAxis') if x.findtext('coordinates')}
        for joint in ['GlenoHumeral','elbow','radioulnar']:
            a,b=axes_by_name(original,joint),axes_by_name(scaled,joint)
            self.assertEqual(a.keys(),b.keys())
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_transplant_anchor_and_existing_capacity(self):
        base=E.parse(INPUT/'scaled_source.osim');reg=E.parse(OUT/'registered_wrist.osim');plus=E.parse(OUT/'registered_plus_elbow.osim');donor=E.parse(INPUT/'donor_scaled.osim')
        delta=elbow_anchor(base)-elbow_anchor(donor)
        for name in WRIST:
            old=base.find(f".//ForceSet/objects/*[@name='{name}']");new=reg.find(f".//ForceSet/objects/*[@name='{name}']")
            self.assertEqual(old.findtext('max_isometric_force'),new.findtext('max_isometric_force'))
            op=[p for p in old.findall('.//PathPointSet/objects/*') if p.findtext('socket_parent_frame')=='/bodyset/humerus']
            npth=[p for p in new.findall('.//PathPointSet/objects/*') if p.findtext('socket_parent_frame')=='/bodyset/humerus']
            for a,b in zip(op,npth):np.testing.assert_allclose(np.fromstring(b.findtext('location'),sep=' ')-np.fromstring(a.findtext('location'),sep=' '),delta,atol=1e-12)
        for name in ELBOW:
            a=donor.find(f".//ForceSet/objects/*[@name='{name}']");b=plus.find(f".//ForceSet/objects/*[@name='{name}']")
            self.assertEqual(a.findtext('max_isometric_force'),b.findtext('max_isometric_force'))
        for body in base.findall('.//BodySet/objects/*'):
            b=plus.find(f".//BodySet/objects/*[@name='{body.get('name')}']")
            for name in ['mass','inertia','mass_center']:self.assertEqual(body.findtext(name),b.findtext(name))
        np.testing.assert_allclose(axes(base),axes(donor),atol=1e-12)

    def test_elbow_moment_arms_match_donor_and_finite_difference(self):
        o.Logger.setLevelString('error');models=[o.Model(str(INPUT/'donor_scaled.osim')),o.Model(str(OUT/'registered_plus_elbow.osim'))];states=[m.initSystem() for m in models]
        for m,s in zip(models,states):
            for n,v in [('elbow_flexion',1.1),('pro_sup',.2)]:m.getCoordinateSet().get(n).setValue(s,v,False)
            m.realizePosition(s)
        for n in ELBOW:
            vals=[m.getMuscles().get(n).computeMomentArm(s,m.getCoordinateSet().get('elbow_flexion')) for m,s in zip(models,states)]
            self.assertAlmostEqual(vals[0],vals[1],places=6)
            m,s=models[1],states[1];c=m.getCoordinateSet().get('elbow_flexion');mus=m.getMuscles().get(n);values=[]
            for v in [1.1-1e-5,1.1+1e-5]:c.setValue(s,v,False);m.realizePosition(s);values.append(mus.getLength(s))
            c.setValue(s,1.1,False);m.realizePosition(s)
            self.assertAlmostEqual(vals[1],-(values[1]-values[0])/2e-5,delta=1e-4)

    def test_pose_unchanged(self):
        rows=json.loads((OUT/'results.json').read_text())['rows']
        for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
            a=[r for r in rows if r['pitch']==pitch]
            self.assertEqual(len(a),3)
            self.assertLess(max(x['arm_rmse_mm'] for x in a)-min(x['arm_rmse_mm'] for x in a),1e-10)

if __name__=='__main__':unittest.main()
