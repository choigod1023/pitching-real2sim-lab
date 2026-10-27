"""Use force production for native model torque, rather than speed inference."""
import numpy as np
import opensim as o
from anatomical_shoulder_fit import arr

def unit_path_force(model,state,path):
    nb=max(model.getBodySet().get(i).getMobilizedBodyIndex() for i in range(model.getBodySet().getSize()))+1
    bodies=o.VectorOfSpatialVec(nb,o.SpatialVec(o.Vec3(0),o.Vec3(0)));forces=o.Vector(state.getNU(),0.)
    path.addInEquivalentForces(state,1.,bodies,forces);body_generalized=o.Vector();model.getMatterSubsystem().multiplyBySystemJacobianTranspose(state,bodies,body_generalized)
    return arr(body_generalized)+arr(forces)

def applied_path_force_matrix(model,state,muscles):
    model.realizeVelocity(state)
    matrix=np.column_stack([unit_path_force(model,state,mus.getPath()) for mus in muscles])
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite native muscle force map')
    return matrix
