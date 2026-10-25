"""Marker-derived kinematic preview, NOT a force-driven pitching simulator."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import ezc3d
import mujoco
from scipy.spatial.transform import Rotation


def frame_rotation(z, lateral):
    z = np.asarray(z, float)
    if np.linalg.norm(z) < 1e-8:
        raise ValueError('Degenerate segment')
    z = z / np.linalg.norm(z)
    y = np.asarray(lateral, float) - np.dot(lateral, z) * z
    if np.linalg.norm(y) < 1e-6:
        axis = np.eye(3)[np.argmin(np.abs(z))]
        y = axis - np.dot(axis, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return Rotation.from_matrix(np.column_stack([x, y, z]))


def wxyz(rotation):
    return np.roll(rotation.as_quat(), 1)


@dataclass
class Trial:
    path: Path
    rate: float
    markers: dict
    joints: dict
    interpolated_samples: int

    @property
    def count(self):
        return len(self.joints['pelvis'])

    @classmethod
    def read(cls, path):
        path = Path(path)
        c = ezc3d.c3d(str(path))
        units = c['parameters']['POINT']['UNITS']['value'][0].strip().lower()
        scale = {'mm': .001, 'm': 1., 'cm': .01}.get(units)
        if scale is None:
            raise ValueError(f'Unsupported C3D units: {units}')
        points = c['data']['points']
        rate = float(c['parameters']['POINT']['RATE']['value'][0])
        labels = c['parameters']['POINT']['LABELS']['value']
        markers, missing = {}, 0
        for i, label in enumerate(labels):
            xyz = points[:3, i].T * scale
            valid = np.isfinite(xyz).all(axis=1) & (points[3, i] >= 0)
            if valid.sum() < 2:
                continue
            missing += int((~valid).sum())
            if not valid.all():
                xyz = np.column_stack([np.interp(np.arange(len(xyz)), np.flatnonzero(valid), xyz[valid, j]) for j in range(3)])
            markers[label.strip()] = xyz
        required = ['LASI', 'RASI', 'LPSI', 'RPSI', 'LSHO', 'RSHO', 'LELB', 'LMELB', 'RELB', 'RMELB', 'LWRA', 'LWRB', 'RWRA', 'RWRB', 'LKNE', 'LMKNE', 'RKNE', 'RMKNE', 'LANK', 'LMANK', 'RANK', 'RMANK']
        absent = set(required) - markers.keys()
        if absent:
            raise ValueError(f'Missing required markers: {sorted(absent)}')
        def avg(*names):
            return np.mean([markers[n] for n in names], axis=0)
        joints = {'pelvis': avg('LASI', 'RASI', 'LPSI', 'RPSI'), 'chest': avg('LSHO', 'RSHO')}
        for side in 'LR':
            joints[side+'hip'] = avg(side+'ASI', side+'PSI')
            joints[side+'shoulder'] = markers[side+'SHO']
            joints[side+'elbow'] = avg(side+'ELB', side+'MELB')
            joints[side+'wrist'] = avg(side+'WRA', side+'WRB')
            joints[side+'knee'] = avg(side+'KNE', side+'MKNE')
            joints[side+'ankle'] = avg(side+'ANK', side+'MANK')
        # Keep laboratory Z, translate horizontal origin to the first pelvis.
        origin = joints['pelvis'][0].copy(); origin[2] = 0
        markers = {k: v-origin for k, v in markers.items()}
        joints = {k: v-origin for k, v in joints.items()}
        return cls(path, rate, markers, joints, missing)


BONES = [('pelvis', 'chest')]
for _s in 'LR':
    BONES += [('chest', _s+'shoulder'), (_s+'shoulder', _s+'elbow'), (_s+'elbow', _s+'wrist'), ('pelvis', _s+'hip'), (_s+'hip', _s+'knee'), (_s+'knee', _s+'ankle')]


class Skeleton:
    """Fixed-length articulated model. Ball joints are visualization joints.

    No muscle, torque, joint-limit, hand/ball, or contact calibration is claimed.
    qpos is imposed per frame and mj_forward is used; mj_step is not used.
    """
    def __init__(self, trial):
        self.trial = trial
        j = trial.joints
        def length(a,b):
            return float(np.median(np.linalg.norm(j[b]-j[a],axis=1)))
        self.lengths = {'spine': length('pelvis','chest')}
        self.widths = {s: {'hip': length('pelvis',s+'hip'), 'shoulder': length('chest',s+'shoulder')} for s in 'LR'}
        for s in 'LR':
            for name,a,b in [('upperarm','shoulder','elbow'),('forearm','elbow','wrist'),('thigh','hip','knee'),('shin','knee','ankle')]:
                self.lengths[s+name] = length(s+a,s+b)
        def segment(name, endpoint, children=''):
            ln=self.lengths[name]
            return f'<joint name="{name}" type="ball"/><geom type="capsule" fromto="0 0 0 0 0 {ln}" size=".025"/><site name="{endpoint}" pos="0 0 {ln}"/>{children}'
        spine=self.lengths['spine']
        arms=''
        legs=''
        for s,sign in [('L',1),('R',-1)]:
            fore=segment(s+'forearm',s+'wrist')
            upper=segment(s+'upperarm',s+'elbow',f'<body pos="0 0 {self.lengths[s+"upperarm"]}">{fore}</body>')
            arms+=f'<body pos="0 {sign*self.widths[s]["shoulder"]} {spine}"><site name="{s}shoulder"/>{upper}</body>'
            shin=segment(s+'shin',s+'ankle')
            thigh=segment(s+'thigh',s+'knee',f'<body pos="0 0 {self.lengths[s+"thigh"]}">{shin}</body>')
            legs+=f'<body pos="0 {sign*self.widths[s]["hip"]} 0"><site name="{s}hip"/>{thigh}</body>'
        shoulder_bar=f'<geom type="capsule" fromto="0 {-self.widths["R"]["shoulder"]} {spine} 0 {self.widths["L"]["shoulder"]} {spine}" size=".025"/>'
        pelvis_bar=f'<geom type="capsule" fromto="0 {-self.widths["R"]["hip"]} 0 0 {self.widths["L"]["hip"]} 0" size=".025"/>'
        torso=segment('spine','chest',shoulder_bar+arms)
        xml=f'''<mujoco model="Pitching - KINEMATIC preview only">
        <visual><global offwidth="1280" offheight="720"/></visual>
        <default><geom rgba=".95 .6 .2 1" contype="0" conaffinity="0"/><site size=".02" rgba="1 .3 .1 1"/></default>
        <worldbody><light pos="0 0 5" ambient=".4 .4 .4"/><geom type="plane" size="5 5 .1" rgba=".14 .17 .2 1"/>
        <body name="pelvis"><freejoint name="root"/><geom type="sphere" size=".06"/><site name="pelvis"/>{pelvis_bar}
        <body>{torso}</body>{legs}</body></worldbody></mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(xml)
        self.data=mujoco.MjData(self.model)
        self.model_xml=xml

    def pose(self, index, display_offset=0.):
        t=self.trial; j={k:v[index] for k,v in t.joints.items()}
        lateral=j['Lhip']-j['Rhip']
        pelvis=frame_rotation([0,0,1],lateral)
        trunk=frame_rotation(j['chest']-j['pelvis'],j['Lshoulder']-j['Rshoulder'])
        self.data.qpos[:3]=j['pelvis']+[0,display_offset,0]
        self.data.qpos[3:7]=wxyz(pelvis)
        def put(name, parent, global_rot):
            address=self.model.jnt_qposadr[self.model.joint(name).id]
            self.data.qpos[address:address+4]=wxyz(parent.inv()*global_rot)
        put('spine',pelvis,trunk)
        for s in 'LR':
            upper=frame_rotation(j[s+'elbow']-j[s+'shoulder'],lateral)
            fore=frame_rotation(j[s+'wrist']-j[s+'elbow'],lateral)
            thigh=frame_rotation(j[s+'knee']-j[s+'hip'],lateral)
            shin=frame_rotation(j[s+'ankle']-j[s+'knee'],lateral)
            for name,parent,rot in [(s+'upperarm',trunk,upper),(s+'forearm',upper,fore),(s+'thigh',pelvis,thigh),(s+'shin',thigh,shin)]:
                put(name,parent,rot)
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)

    def residuals(self,index):
        self.pose(index)
        return {k:float(np.linalg.norm(self.data.site(k).xpos-v[index])) for k,v in self.trial.joints.items()}


def draw_reference(scene, trial, index, offset=-1.3):
    scene.ngeom=0
    shift=np.array([0,offset,0])
    for a,b in BONES:
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.2,.85,.75,1.]))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.016,trial.joints[a][index]+shift,trial.joints[b][index]+shift)
        scene.ngeom+=1
    for xyz in trial.markers.values():
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009,0,0]),xyz[index]+shift,np.eye(3).ravel(),np.array([.4,1,.8,1.]))
        scene.ngeom+=1
