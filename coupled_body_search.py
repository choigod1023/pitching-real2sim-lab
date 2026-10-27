"""Reduced coupled-body mechanics, with explicit unvalidated pelvis support."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, RotationSpline
from full_body_comparison import load
from pilot import regular_time,vector
from torque_arm import arm_poses,derivatives,unit
from coupled_payload import model_for,wxyz
from capacity_budget_gate import split_linear_power

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/coupled-body'

def rot(q):return R.from_quat(np.asarray(q)[...,[1,2,3,0]])
def quat(r):return wxyz(r)
def frame(x,z):
    z=unit(z);x=unit(x-(x*z).sum(1)[:,None]*z);y=np.cross(z,x)
    return R.from_matrix(np.stack([x,y,z],axis=-1))
def txt(a):return ' '.join(map(str,a))

def build(dt_target=1/2880,pitch="3034_4",observed_pitch=None,padding=0,output_dir=None,frame_policy="segment_axes",reconstruct_rounding=False):
    output=OUT if output_dir is None else Path(output_dir)
    meta,poi,tables=load();g=tables['landmarks'].query('session_pitch == @pitch').reset_index(drop=True)
    if reconstruct_rounding:
        from quantization_reconstruction import reconstruct_landmarks
        g,_=reconstruct_landmarks(g)
    native=regular_time(g);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0]);n=round((br-fp)/dt_target);dt=(br-fp)/n;t=fp+np.arange(-padding,n+padding+1)*dt
    pts={name:CubicSpline(native,vector(g,name))(t) for name in ['rear_hip','lead_hip','rear_knee_jc','lead_knee_jc','rear_ankle_jc','lead_ankle_jc','shoulder_jc','glove_shoulder_jc','glove_elbow_jc','glove_wrist_jc']}
    center=(pts['rear_hip']+pts['lead_hip'])/2;top=(pts['shoulder_jc']+pts['glove_shoulder_jc'])/2
    pelvis=frame(pts['rear_hip']-pts['lead_hip'],np.tile([0.,0.,1.],(len(t),1)))
    torso=frame(pts['shoulder_jc']-pts['glove_shoulder_jc'],top-center)
    if frame_policy in ('marker_axes','marker_axes_lumbar','segmented_trunk'):
        if observed_pitch is None:raise ValueError('Marker frames require source observation')
        import ezc3d
        from scipy.signal import butter,sosfiltfilt
        c=ezc3d.c3d(observed_pitch.ledger['source_paths']['c3d']);labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        rate=float(c['parameters']['POINT']['RATE']['value'][0]);scale={'mm':.001,'cm':.01,'m':1.}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
        markers={}
        for name in ['RASI','LASI','RPSI','LPSI']:
            raw=c['data']['points'][:,labels.index(name),:].T
            if not np.isfinite(raw).all() or np.any(raw[:,3]<0):raise ValueError('Missing pelvis markers')
            filtered=sosfiltfilt(butter(4,20,fs=rate,output='sos'),raw[:,:3]*scale,axis=0)
            markers[name]=CubicSpline(np.arange(len(filtered))/rate,filtered)(t)
        px=markers['RASI']-markers['LASI'];py=(markers['RASI']+markers['LASI']-markers['RPSI']-markers['LPSI'])/2
        pz=np.cross(px,py)
        if np.mean(pz[:,2])<0:pz=-pz
        pelvis=frame(px,pz)
        proximal=CubicSpline(native,vector(g,'thorax_prox'))(t);distal=CubicSpline(native,vector(g,'thorax_dist'))(t)
        torso=frame(pts['shoulder_jc']-pts['glove_shoulder_jc'],proximal-distal)
    elif frame_policy!='segment_axes':raise ValueError('Unknown frame policy')
    lumbar=np.zeros(3)
    if frame_policy=='marker_axes_lumbar':
        lumbar=np.mean(pelvis.inv().apply(distal-center),axis=0)
    trunk_origin=center+pelvis.apply(np.tile(lumbar,(len(t),1)))
    if frame_policy=='segmented_trunk':
        lumbar_length=float(np.median(np.linalg.norm(distal-center,axis=1)))
        lumbar_rotation=frame(pelvis.as_matrix()[:,:,0],distal-center)
        trunk_origin=center+lumbar_rotation.apply(np.tile([0,0,lumbar_length],(len(t),1)))
    shoulder_local=np.median(torso.inv().apply(pts['shoulder_jc']-trunk_origin),axis=0)
    glove_local=np.median(torso.inv().apply(pts['glove_shoulder_jc']-trunk_origin),axis=0)
    capsule_start=np.median(torso.inv().apply(center-trunk_origin),axis=0)
    capsule_end=np.median(torso.inv().apply(top-trunk_origin),axis=0)
    if frame_policy=='segmented_trunk':capsule_start=np.zeros(3)
    height=float(np.median(np.linalg.norm(top-center,axis=1)));width=float(np.median(np.linalg.norm(pts['shoulder_jc']-pts['glove_shoulder_jc'],axis=1)))
    if observed_pitch is not None:
        aq=observed_pitch.sample(t);arm_lengths=observed_pitch.lengths
    else:
        if pitch!='3034_4' or padding:raise ValueError('ObservedPitch required for other pitches or padding')
        source=ROOT.parent/'trace-error-audit/artifacts/trace-error/3034_4-46080-pad1-foh-sub1.npz'
        z=np.load(source);at=np.linspace(fp,br,len(z['qreference']));aq=np.zeros((len(t),13));aq[:,:3]=CubicSpline(at,z['qreference'][:,:3])(t)
        for sl in [slice(3,7),slice(9,13)]:aq[:,sl]=quat(RotationSpline(at,rot(z['qreference'][:,sl]))(t))
        aq[:,7:9]=CubicSpline(at,z['qreference'][:,7:9])(t)
        arm_lengths=z['lengths']
    # Serialize inherited arm model so mass/grip and all hand DOFs remain unchanged.
    output.mkdir(parents=True,exist_ok=True)
    knee_native=arm_poses(vector(g,'rear_hip'),vector(g,'rear_knee_jc'),vector(g,'rear_ankle_jc'))[:,7]
    np.savez(output/'knee_source.npz',t=native-fp,angle=knee_native)
    marm=model_for(arm_lengths,dt);mujoco.mj_saveLastXML(str(output/'arm.xml'),marm)
    tree=ET.parse(output/'arm.xml');xml=tree.getroot();world=xml.find('worldbody');upper=world.find('body');world.remove(upper)
    free=upper.find('freejoint')
    if free is None:free=upper.find("joint[@type='free']")
    upper.remove(free);ET.SubElement(upper,'joint',name='pitch_shoulder',type='ball')
    upper.set('pos',txt(shoulder_local if frame_policy in ('marker_axes_lumbar','segmented_trunk') else [width/2,0,height]));upper.set('name','upper')
    masses=json.loads((ROOT/'body_mass_source.json').read_text())['mass_kg']
    root=ET.SubElement(world,'body',name='pelvis');ET.SubElement(root,'freejoint',name='root_support')
    ET.SubElement(root,'geom',type='box',size='.12 .10 .09',mass=str(masses['pelvis']))
    trunk_parent=root;trunk_offset=lumbar;trunk_mass=masses['torso']
    if frame_policy=='segmented_trunk':
        lower_mass=trunk_mass*lumbar_length/(lumbar_length+np.linalg.norm(capsule_end));trunk_mass-=lower_mass
        trunk_parent=ET.SubElement(root,'body',name='lower_trunk');ET.SubElement(trunk_parent,'joint',name='lower_trunk_rotation',type='ball')
        ET.SubElement(trunk_parent,'geom',type='capsule',fromto=txt([0,0,0,0,0,lumbar_length]),size='.13',mass=str(lower_mass))
        trunk_offset=[0,0,lumbar_length]
    trunk=ET.SubElement(trunk_parent,'body',name='torso',pos=txt(trunk_offset));ET.SubElement(trunk,'joint',name='torso_rotation',type='ball')
    ET.SubElement(trunk,'geom',type='capsule',fromto=txt(np.r_[capsule_start,capsule_end] if frame_policy in ('marker_axes_lumbar','segmented_trunk') else [0,0,0,0,0,height]),size='.13',mass=str(trunk_mass))
    trunk.append(upper)
    qmap={'root_support':np.c_[center,quat(pelvis)],'torso_rotation':quat(pelvis.inv()*torso),
          'pitch_shoulder':quat(torso.inv()*rot(aq[:,3:7])),'elbow':aq[:,7:8],'marker_roll':aq[:,8:9],'wrist_marker_rotation':aq[:,9:13]}
    if frame_policy=='segmented_trunk':
        qmap['lower_trunk_rotation']=quat(pelvis.inv()*lumbar_rotation);qmap['torso_rotation']=quat(lumbar_rotation.inv()*torso)
    sites={}
    for side,sidecode in [('rear','r'),('lead','l')]:
        hip=pts[side+'_hip'];knee=pts[side+'_knee_jc'];ankle=pts[side+'_ankle_jc'];p=arm_poses(hip,knee,ankle)
        lengths=[float(np.median(np.linalg.norm(knee-hip,axis=1))),float(np.median(np.linalg.norm(ankle-knee,axis=1)))]
        loc=np.median(pelvis.inv().apply(hip-center),axis=0)
        thigh=ET.SubElement(root,'body',name=side+'_thigh',pos=txt(loc));ET.SubElement(thigh,'joint',name=side+'_hip',type='ball')
        ET.SubElement(thigh,'geom',type='capsule',fromto=txt([0,0,0,0,0,-lengths[0]]),size='.06',mass=str(masses['femur_'+sidecode]+masses['patella_'+sidecode]))
        shank=ET.SubElement(thigh,'body',name=side+'_shank',pos=txt([0,0,-lengths[0]]));ET.SubElement(shank,'joint',name=side+'_knee',type='hinge',axis='0 1 0')
        ET.SubElement(shank,'geom',type='capsule',fromto=txt([0,0,0,0,0,-lengths[1]]),size='.04',mass=str(masses['tibia_'+sidecode]))
        ET.SubElement(shank,'geom',type='sphere',pos=txt([0,0,-lengths[1]]),size='.05',mass=str(sum(masses[x+'_'+sidecode] for x in ['talus','calcn','toes'])))
        for body,name,pos in [(thigh,side+'_hip',[0,0,0]),(shank,side+'_knee_jc',[0,0,0]),(shank,side+'_ankle_jc',[0,0,-lengths[1]])]:ET.SubElement(body,'site',name=name,pos=txt(pos))
        qmap[side+'_hip']=quat(pelvis.inv()*rot(p[:,3:7]));qmap[side+'_knee']=p[:,7:8]
    # Glove forearm and hand are a single rigid load, retaining observed elbow motion.
    gp=arm_poses(pts['glove_shoulder_jc'],pts['glove_elbow_jc'],pts['glove_wrist_jc'])
    lengths=[float(np.median(np.linalg.norm(pts[b]-pts[a],axis=1))) for a,b in [('glove_shoulder_jc','glove_elbow_jc'),('glove_elbow_jc','glove_wrist_jc')]]
    gu=ET.SubElement(trunk,'body',name='glove_upper',pos=txt(glove_local if frame_policy in ('marker_axes_lumbar','segmented_trunk') else [-width/2,0,height]));ET.SubElement(gu,'joint',name='glove_shoulder',type='ball')
    ET.SubElement(gu,'geom',type='capsule',fromto=txt([0,0,0,0,0,-lengths[0]]),size='.04',mass=str(masses['humerus_l']))
    gf=ET.SubElement(gu,'body',name='glove_forearm',pos=txt([0,0,-lengths[0]]));ET.SubElement(gf,'joint',name='glove_elbow',axis='0 1 0')
    ET.SubElement(gf,'geom',type='capsule',fromto=txt([0,0,0,0,0,-lengths[1]]),size='.03',mass=str(masses['ulna_l']+masses['radius_l']+masses['hand_l']))
    qmap['glove_shoulder']=quat(torso.inv()*rot(gp[:,3:7]));qmap['glove_elbow']=gp[:,7:8]
    for body,name,pos in [(upper,'shoulder_jc',[0,0,0]),(upper.find('body'),'elbow_jc',[0,0,0]),(gu,'glove_shoulder_jc',[0,0,0]),(gf,'glove_elbow_jc',[0,0,0]),(gf,'glove_wrist_jc',[0,0,-lengths[1]])]:ET.SubElement(body,'site',name=name,pos=txt(pos))
    xml.find('option').set('integrator','RK4');tree.write(output/'model.xml')
    m=mujoco.MjModel.from_xml_path(str(output/'model.xml'));q=np.zeros((len(t),m.nq))
    for name,value in qmap.items():
        adr=int(m.joint(name).qposadr[0]);q[:,adr:adr+value.shape[1]]=value
    return m,q,dt,t,pts

def modify(m,q,dt,delay_ms,angle_deg,knee_source=None):
    out=q.copy();t=np.arange(len(q))*dt;u=np.clip((t-.015)/.04,0,1);r=u**3*(10-15*u+6*u*u)
    # Shift the observed knee trajectory, including available samples after BR.
    # This avoids forcing the time map back at BR and distorting its peak.
    adr=int(m.joint('rear_knee').qposadr[0]);shift=t-delay_ms*.001*r
    z=np.load(OUT/'knee_source.npz') if knee_source is None else knee_source
    if shift.min()<min(z['t']) or shift.max()>max(z['t']):raise ValueError('Retiming extrapolation')
    spline=CubicSpline(z['t'],z['angle'])
    out[:,adr]=q[:,adr]+spline(shift)-spline(t)
    adr=int(m.joint('torso_rotation').qposadr[0]);rv=np.zeros((len(q),3));rv[:,2]=-np.deg2rad(angle_deg)*r
    out[:,adr:adr+4]=quat(R.from_rotvec(rv)*rot(q[:,adr:adr+4]))
    return out

def groups(m):
    out={'external_force':np.arange(3),'external_torque':np.arange(3,6)}
    for j in range(1,m.njnt):
        a=m.jnt_dofadr[j];b=m.jnt_dofadr[j+1] if j+1<m.njnt else m.nv;out[m.joint(j).name]=np.arange(a,b)
    return out

def inverse(m,q,dt):
    v,a=derivatives(m,q,dt);d=mujoco.MjData(m);f=[]
    for p,vv,aa in zip(q,v,a):
        d.qpos[:]=p;d.qvel[:]=vv;d.qacc[:]=aa;mujoco.mj_inverse(m,d);f.append(d.qfrc_inverse.copy())
    return np.array(f),v

def simulate(m,q,forces,v0,dt,sub=1):
    m.opt.timestep=dt/sub;d=mujoco.MjData(m);d.qpos[:]=q[0];d.qvel[:]=v0;mujoco.mj_forward(m,d)
    gs=groups(m);pos=np.zeros(len(gs));neg=pos.copy();states=[d.qpos.copy()];velocities=[d.qvel.copy()]
    initial=(d.qpos.copy(),d.qvel.copy());mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);e0=d.energy.sum()
    for force in forces[:-1]:
        d.qfrc_applied[:]=force
        for k in range(sub):
            p0=np.array([force[ix]@d.qvel[ix] for ix in gs.values()]);mujoco.mj_step(m,d)
            p1=np.array([force[ix]@d.qvel[ix] for ix in gs.values()]);pp,nn,_=split_linear_power(p0,p1,m.opt.timestep);pos+=pp;neg+=nn
            if d.warning.number.any() or not np.isfinite(d.qvel).all():raise RuntimeError('Unstable simulation')
        states.append(d.qpos.copy());velocities.append(d.qvel.copy())
    mujoco.mj_forward(m,d);jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
    mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
    costs={name:dict(positive_j=float(pos[i]),negative_j=float(neg[i]),peak=float(np.linalg.norm(forces[:-1,ix],axis=1).max())) for i,(name,ix) in enumerate(gs.items())}
    return dict(speed_kmh=float(np.linalg.norm(jp@d.qvel)*3.6),costs=costs,energy_residual_j=float(d.energy.sum()-e0-(pos-neg).sum())),np.array(states),np.array(velocities)

def positions(m,q):
    names=['rear_hip','lead_hip','rear_knee_jc','lead_knee_jc','rear_ankle_jc','lead_ankle_jc','shoulder_jc','elbow_jc','wrist','ball_center','glove_shoulder_jc','glove_elbow_jc','glove_wrist_jc']
    d=mujoco.MjData(m);p=[]
    for x in q:
        d.qpos[:]=x;mujoco.mj_forward(m,d);p.append([d.site_xpos[m.site(n).id].copy() for n in names])
    return names,np.array(p)

def compare(m,base,result,bq,cq):
    violations=[]
    for name,c in result['costs'].items():
        for metric,value in c.items():
            excess=value-base['costs'][name][metric]
            if excess>1e-8:violations.append(dict(group=name,metric=metric,excess=excess))
    names,b=positions(m,bq);_,c=positions(m,cq);dq=np.zeros(m.nv);deltas=[]
    for x,y in zip(bq,cq):mujoco.mj_differentiatePos(m,dq,1,x,y);deltas.append(dq.copy())
    diff=np.array(deltas);maxangle=max(np.linalg.norm(diff[:,ix],axis=1).max()*180/np.pi for name,ix in groups(m).items() if name!='external_force')
    ankle=max(np.linalg.norm(c[:,names.index(n)]-b[:,names.index(n)],axis=1).max()*1000 for n in ['rear_ankle_jc','lead_ankle_jc'])
    ball=np.linalg.norm(c[:,names.index('ball_center')]-b[:,names.index('ball_center')],axis=1).max()*1000
    return dict(gain_kmh=result['speed_kmh']-base['speed_kmh'],violations=violations,budget_pass=not violations,max_joint_change_deg=float(maxangle),max_ankle_change_mm=float(ankle),max_ball_change_mm=float(ball),pose_pass=bool(maxangle<=3 and ankle<=20 and ball<=20),athlete_candidate_validated=False)

def run():
    m,q,dt,t,observed=build();f,v=inverse(m,q,dt);bases={};runs=[];saved={}
    for sub in [1,2]:bases[sub]=simulate(m,q,f,v[0],dt,sub)
    for delay in [-8.,-4.,0.,4.,8.]:
        for angle in [-2.37,-1.,0.,1.,2.37]:
            changed=modify(m,q,dt,delay,angle);force,cv=inverse(m,changed,dt)
            np.testing.assert_allclose(changed[:4],q[:4],atol=1e-14)
            for sub in [1,2]:
                r,p,vel=simulate(m,changed,force,v[0],dt,sub);base,bp,bv=bases[sub]
                r.update(compare(m,base,r,bp,p));r.update(delay_parameter_ms=delay,separation_parameter_deg=angle,substeps=sub)
                knee=int(m.joint('rear_knee').dofadr[0]);r['actual_knee_peak_shift_ms']=float((np.argmax(np.abs(vel[:,knee]))-np.argmax(np.abs(bv[:,knee])))*dt*1000)
                runs.append(r)
                if sub==2:saved[(delay,angle)]=(p,vel,force,changed)
    fine=[r for r in runs if r['substeps']==2];nonzero=[r for r in fine if r['delay_parameter_ms'] or r['separation_parameter_deg']]
    best=max(nonzero,key=lambda r:r['gain_kmh']);key=(best['delay_parameter_ms'],best['separation_parameter_deg']);p,vel,force,changed=saved[key]
    refnames,ref=positions(m,q);_,bp=positions(m,bases[2][1]);fit_errors={n:float(np.sqrt(np.mean(np.sum((ref[:,refnames.index(n)]-observed[n])**2,axis=1)))*1000) for n in refnames if n in observed};rmse=float(np.sqrt(np.mean(np.sum((ref-bp)**2,axis=2)))*1000)
    coarse=next(r for r in runs if r['substeps']==1 and (r['delay_parameter_ms'],r['separation_parameter_deg'])==key)
    fine4,p4,v4=simulate(m,changed,force,v[0],dt,4);base4,b4,bv4=simulate(m,q,f,v[0],dt,4)
    # Also halve the input sampling interval; integration subdivision alone is insufficient.
    mh,qh,dth,th,_=build(dt/2);fh,vh=inverse(mh,qh,dth);bh,bhp,bhv=simulate(mh,qh,fh,vh[0],dth,2)
    ch=modify(mh,qh,dth,*key);cfh,_=inverse(mh,ch,dth);rh,ph,vph=simulate(mh,ch,cfh,vh[0],dth,2);rh.update(compare(mh,bh,rh,bhp,ph))
    np.savez_compressed(OUT/'overlay_states.npz',time=t,qbaseline=bases[2][1],qcandidate=p,vbaseline=bases[2][2],vcandidate=vel,reference=q,dt=dt)
    # restore original-step model XML for reproducible video/state dimensional mapping
    m,_,_,_,_=build(dt)
    report=dict(pitch='3034_4',measured_speed_kmh=79.8*1.609344,baseline=bases[2][0],best_diagnostic=best,
                nonzero_conditions=len(nonzero),nonzero_budget_pass=sum(r['budget_pass'] for r in nonzero),
                positive_budget_pose_pass=sum(r['budget_pass'] and r['pose_pass'] and r['gain_kmh']>0 for r in nonzero),
                validated_athlete_candidates=0,reference_reproduction_rmse_mm=rmse,observed_landmark_fit_rmse_mm=fit_errors,model_mass_including_ball_kg=float(m.body_mass.sum()),
                input_sampling_refinement=dict(dt=dth,baseline_speed=bh['speed_kmh'],diagnostic=rh),
                subdivision_check=dict(coarse_gain=coarse['gain_kmh'],fine_gain=best['gain_kmh'],four_sub_gain=fine4['speed_kmh']-base4['speed_kmh']),
                runs=runs,ground_support_validated=False,muscles_validated=False,
                note='Nominal mass rigid-body demand test with floating pelvis external wrench; no ground contact model. Post-FP separation perturbation only, not FP-angle intervention. No physiological or athlete gain claim.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['runs','baseline','best_diagnostic','input_sampling_refinement']},indent=2))
    print('best',key,best['speed_kmh'],best['gain_kmh'],'failures',best['violations']);return report

if __name__=='__main__':run()
