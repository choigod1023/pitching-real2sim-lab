"""Kinematic proximity to a defined sweet-spot zone, without impact response."""
import numpy as np
from ..observed_motion.overlay import interpolate

GRAVITY = np.array([0., 0., -9.80665])


def point_segment_distance(point, a, b):
    p, a, b = np.broadcast_arrays(np.asarray(point,float), np.asarray(a,float), np.asarray(b,float))
    if p.shape[-1] != 3 or not all(np.isfinite(v).all() for v in (p,a,b)):
        raise ValueError('invalid_geometry')
    delta=b-a;den=np.sum(delta*delta,axis=-1)
    ratio=np.divide(np.sum((p-a)*delta,axis=-1),den,out=np.zeros_like(den),where=den>1e-20)
    closest=a+np.clip(ratio,0,1)[...,None]*delta
    return np.linalg.norm(p-closest,axis=-1)


def ballistic_initial(origin, destination, speed):
    origin,destination=np.asarray(origin,float),np.asarray(destination,float)
    if (origin.shape!=(3,) or destination.shape!=(3,) or not np.isfinite(origin).all()
        or not np.isfinite(destination).all() or not np.isfinite(speed) or speed<=0):
        raise ValueError('invalid_projectile_input')
    d=destination-origin;a=.25*float(GRAVITY@GRAVITY);b=-float(d@GRAVITY+speed*speed);c=float(d@d)
    disc=b*b-4*a*c
    if disc<0 or c<=0 or -b+np.sqrt(max(disc,0))<=0:raise ValueError('unreachable_destination')
    time=np.sqrt(2*c/(-b+np.sqrt(disc)))
    velocity=(d-.5*GRAVITY*time*time)/time
    return velocity,float(time)


def ball_positions(origin, velocity, time):
    origin,velocity=np.asarray(origin,float),np.asarray(velocity,float)
    time=np.asarray(time,float)
    if (origin.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(origin).all()
        or not np.isfinite(velocity).all() or not np.isfinite(time).all() or np.any(time<0)):
        raise ValueError('invalid_projectile_time_or_state')
    return np.asarray(origin)+time[...,None]*np.asarray(velocity)+.5*time[...,None]**2*GRAVITY


def bat_state(track, scene_time, arrival, delay_ms):
    query=track['contact']+np.asarray(scene_time)-arrival-delay_ms/1000.
    grip=interpolate(track['time'],track['grip'],query)
    sweet=interpolate(track['time'],track['sweet'],query)
    direction=sweet-grip;length=np.linalg.norm(direction,axis=-1)
    if np.any(length<.1):raise ValueError('degenerate_hand_sweet_axis')
    return grip,sweet,direction/length[...,None]


def proximity(track, origin, velocity, arrival, delay_ms, *, step=.0001, bat_radius=.033, z_offset=0.):
    if not np.isfinite(step) or step<=0 or not np.isfinite(bat_radius) or bat_radius<=0:
        raise ValueError('invalid_geometry_assumptions')
    start=max(0.,arrival-.06,arrival+delay_ms/1000+track['time'][0]-track['contact'])
    stop=min(arrival+.06,arrival+delay_ms/1000+track['time'][-1]-track['contact'])
    if stop<=start:raise ValueError('no_common_time_support')
    times=np.arange(start,stop+step*.1,step);times=times[times<=stop]
    _,sweet,axis=bat_state(track,times,arrival,delay_ms)
    ball=ball_positions(origin,velocity,times)+np.array([0.,0.,z_offset])
    distance=point_segment_distance(ball,sweet-.05*axis,sweet+.05*axis)
    clearance=distance-(.0365+bat_radius);k=int(np.argmin(clearance))
    return dict(delay_ms=float(delay_ms),minimum_clearance_mm=float(clearance[k]*1000),
        closest_time_from_nominal_contact_ms=float((times[k]-arrival)*1000),
        sweet_zone_near=bool(clearance[k]<=0),step_ms=step*1000,
        measured_contact_or_miss=False)


def sweep(track, origin, velocity, arrival, *, step=.0001, bat_radius=.033, z_offset=0.):
    cases=[proximity(track,origin,velocity,arrival,delay,step=step,bat_radius=bat_radius,z_offset=z_offset)
           for delay in range(-20,21)]
    intervals=[];begin=None
    for case in cases+[dict(delay_ms=21,sweet_zone_near=False)]:
        if case['sweet_zone_near'] and begin is None:begin=case['delay_ms']
        elif not case['sweet_zone_near'] and begin is not None:
            intervals.append([begin,case['delay_ms']-1]);begin=None
    return dict(cases=cases,grid_near_intervals_ms=intervals,
        largest_sampled_interval_width_ms=max((b-a for a,b in intervals),default=0),
        touches_sweep_boundary=cases[0]['sweet_zone_near'] or cases[-1]['sweet_zone_near'],
        meaning='conditional_sweet_zone_proximity_not_whiff_probability')
