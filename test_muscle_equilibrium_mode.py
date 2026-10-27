import unittest
import numpy as np
import opensim as o
from anatomical_shoulder_fit import OUT
from anatomical_muscle_screen import equilibrium_tendon_forces

class EquilibriumModeTests(unittest.TestCase):
    def test_disabled_force_state_does_not_skip_equilibrium(self):
        o.Logger.setLevelString('error');m=o.Model(str(OUT/'scaled_source.osim'));s=m.initSystem()
        z=np.load(OUT/'3034_4_ik.npz')
        for j in range(m.getNumCoordinates()):m.getCoordinateSet().get(j).setValue(s,float(z['q'][30,j]),False)
        muscles=[m.getMuscles().get(i) for i in range(33)]
        for mus in muscles:mus.setAppliesForce(s,False);mus.setActivation(s,0.)
        m.equilibrateMuscles(s);m.realizeDynamics(s)
        old=m.getMuscles().get('Rhomboideus_I').getTendonForce(s)
        forces,lengths=equilibrium_tendon_forces(m,s,muscles,0.)
        i=[x.getName() for x in muscles].index('Rhomboideus_I')
        self.assertGreater(old,10000.)
        self.assertLess(forces[i],100.)
        self.assertTrue(all(not x.appliesForce(s) for x in muscles))
        self.assertTrue(np.isfinite(forces).all())
        # Independent enabled equilibrium agrees; this was the regression.
        for mus in muscles:mus.setAppliesForce(s,True)
        m.equilibrateMuscles(s);m.realizeDynamics(s)
        np.testing.assert_allclose(forces,[x.getTendonForce(s) for x in muscles],rtol=1e-4,atol=1e-4)

if __name__=='__main__':unittest.main()
