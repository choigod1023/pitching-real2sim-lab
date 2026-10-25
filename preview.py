"""Launch native MuJoCo playback with a Korean Tk control panel."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from kinematics import Trial, Skeleton, draw_reference


def audit(skeleton):
    rows=[skeleton.residuals(i) for i in range(skeleton.trial.count)]
    values=np.array([list(row.values()) for row in rows])*1000
    return {'mode':'kinematic_only', 'trial':skeleton.trial.path.name,
            'frames':skeleton.trial.count,'rate_hz':skeleton.trial.rate,
            'interpolated_marker_samples':skeleton.trial.interpolated_samples,
            'mean_endpoint_distance_mm':float(values.mean()),
            'p95_endpoint_distance_mm':float(np.percentile(values,95)),
            'max_endpoint_distance_mm':float(values.max()),
            'per_site_mean_mm':dict(zip(rows[0],values.mean(axis=0).tolist())),
            'segment_lengths_m':skeleton.lengths,
            'limitations':['Marker-derived joint proxies, not calibrated anatomical joint centers',
                           'Fixed lengths and generic ball joints; no anatomical limits',
                           'qpos imposed; no force-driven simulation or pitching controller',
                           'No ball release, predicted velocity, comfort or injury claims']}


def render(skeleton, index, path):
    from PIL import Image
    skeleton.pose(index,1.3)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.5,0,1.]
    camera.distance=5.4; camera.azimuth=0; camera.elevation=-15
    with mujoco.Renderer(skeleton.model,720,1280) as renderer:
        renderer.update_scene(skeleton.data,camera=camera)
        # Preserve MuJoCo's model geoms and append the reference geometry.
        scene=mujoco.MjvScene(skeleton.model,maxgeom=200)
        draw_reference(scene,skeleton.trial,index)
        target=renderer.scene
        for i in range(scene.ngeom):
            src=scene.geoms[i]; dst=target.geoms[target.ngeom]
            mujoco.mjv_initGeom(dst,src.type,src.size.astype(float),src.pos.astype(float),src.mat.astype(float).ravel(),src.rgba)
            # MjvGeom capsule sizes are expanded (r,r,half-length), while
            # mjv_initGeom accepts model sizes (r,half-length,unused).
            dst.size[:]=src.size
            dst.emission=.25
            target.ngeom+=1
        Image.fromarray(renderer.render()).save(path)


def launch(skeleton, report, smoke_seconds=0):
    import tkinter as tk
    from tkinter import ttk
    import mujoco.viewer
    root=tk.Tk(); root.title('투구 연구실 | 실제 측정 ↔ 관절 모델')
    root.geometry('660x510'); root.minsize(620,500)
    style=ttk.Style(root); style.configure('TLabel',font=('Malgun Gothic',10))
    style.configure('TButton',font=('Malgun Gothic',10))
    frame=ttk.Frame(root,padding=18); frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='실제 투구 재현 · 첫 번째 연구 단계',font=('Malgun Gothic',16,'bold')).pack(anchor='w')
    ttk.Label(frame,text='좌표 재생 모드 — 힘으로 생성한 투구 시뮬레이션이 아닙니다.',foreground='#a04b00').pack(anchor='w',pady=(8,12))
    ttk.Label(frame,text='청록색: 실제 측정 마커와 연결선\n주황색: 고정된 뼈 길이를 갖는 단순 관절 모델').pack(anchor='w')
    ttk.Label(frame,text=f'파일: {skeleton.trial.path.name}',wraplength=610).pack(anchor='w',pady=12)
    status=tk.StringVar(); ttk.Label(frame,textvariable=status).pack(anchor='w')
    position=tk.DoubleVar(value=0)
    playing=[False]; clock=[time.perf_counter()]; updating=[False]
    def seek(value):
        if not updating[0]:
            playing[0]=False
    slider=ttk.Scale(frame,from_=0,to=skeleton.trial.count-1,variable=position,command=seek)
    slider.pack(fill='x',pady=10)
    controls=ttk.Frame(frame); controls.pack(fill='x')
    def toggle():
        playing[0]=not playing[0]; clock[0]=time.perf_counter()
    def step(delta):
        playing[0]=False; position.set(np.clip(round(position.get())+delta,0,skeleton.trial.count-1))
    ttk.Button(controls,text='재생 / 일시정지',command=toggle).pack(side='left')
    ttk.Button(controls,text='◀ 1 프레임',command=lambda:step(-1)).pack(side='left',padx=5)
    ttk.Button(controls,text='1 프레임 ▶',command=lambda:step(1)).pack(side='left')
    speed=tk.StringVar(value='0.25')
    ttk.Label(controls,text='  배속 ').pack(side='left')
    ttk.Combobox(controls,textvariable=speed,values=['0.1','0.25','0.5','1.0'],state='readonly',width=5).pack(side='left')
    ttk.Label(frame,text=f'측정 {skeleton.trial.rate:.0f}Hz · {skeleton.trial.count}프레임\n관절 대리점 거리: 평균 {report["mean_endpoint_distance_mm"]:.1f}mm / 95백분위 {report["p95_endpoint_distance_mm"]:.1f}mm\n이 수치는 자세 추정 정확도가 아니라 단순 모델과 측정 위치의 차이입니다.',wraplength=610).pack(anchor='w',pady=15)
    ttk.Label(frame,text='마우스 드래그: 시점 회전 · 휠: 확대/축소\n프레임 이동과 재생은 이 제어창에서 조작하세요.\n현재 단계에서는 구속 향상·필요한 힘·편안함을 계산하지 않습니다.',wraplength=610).pack(anchor='w')
    skeleton.pose(0,1.3)
    viewer=mujoco.viewer.launch_passive(skeleton.model,skeleton.data)
    with viewer.lock():
        viewer.cam.lookat[:]=[.5,0,1.]; viewer.cam.distance=5.4
        viewer.cam.azimuth=0; viewer.cam.elevation=-15
    closed=[False]
    def close():
        if closed[0]: return
        closed[0]=True; viewer.close(); root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    def update():
        if closed[0]: return
        if not viewer.is_running(): close(); return
        now=time.perf_counter(); dt=now-clock[0]; clock[0]=now
        if playing[0]:
            updating[0]=True
            position.set((position.get()+dt*skeleton.trial.rate*float(speed.get()))%skeleton.trial.count)
            updating[0]=False
        index=min(skeleton.trial.count-1,int(position.get()))
        with viewer.lock():
            skeleton.pose(index,1.3)
            draw_reference(viewer.user_scn,skeleton.trial,index)
        viewer.sync()
        status.set(f'프레임 {index+1} / {skeleton.trial.count}     시간 {index/skeleton.trial.rate:.3f}초')
        root.after(16,update)
    update()
    if smoke_seconds:
        playing[0]=True
        root.after(int(smoke_seconds*1000),close)
    root.mainloop()
    print('GUI closed normally',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--trial',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-seconds',type=float,default=0)
    args=parser.parse_args()
    if args.trial:
        path=args.trial
    else:
        paths=sorted(p for p in (args.data_root/'c3d').rglob('*.c3d') if 'model' not in p.name.lower())
        if not paths: parser.error('C3D data missing; pass --data-root or --trial')
        path=paths[0]
    skeleton=Skeleton(Trial.read(path)); report=audit(skeleton)
    out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'preview-model.xml').write_text(skeleton.model_xml,encoding='utf-8')
    render(skeleton,int(skeleton.trial.count*.65),out/'preview.png')
    print(json.dumps(report,indent=2),flush=True)
    if not args.check_only: launch(skeleton,report,args.smoke_seconds)


if __name__=='__main__':main()
