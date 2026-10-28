"""Local desktop interface for conditional joint-to-flight comparisons."""
import json
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from pitch_outcomes import evaluate


def launch(data, output):
    root = tk.Tk(); root.title('야구 관절–공 비행 가설 실험'); root.geometry('1160x830')
    ttk.Label(root, text='관절 수정 → 실제 팔 적분 → 가정한 공 비행 | 선수 향상 검증 전', font=('Malgun Gothic', 16)).pack(pady=8)
    ttk.Label(root, text='회전·목표점·재현 편차는 가정입니다. 목표점 오차 감소 ≠ 실제 제구 향상. 회전 생성과 근력 능력은 미검증.').pack()
    controls = ttk.Frame(root); controls.pack(pady=8); entries = {}
    for col, (key, label, value) in enumerate([('x','목표 좌우(m)',0),('z','목표 높이(m)',.9),
        ('cl','유효 양력계수',.15),('cd','항력계수',.35),('jitter','방향 편차 SD(도)',.15),('offset','공통 방향 가정(도)',0)]):
        ttk.Label(controls,text=label).grid(row=0,column=col)
        entry=ttk.Entry(controls,width=15); entry.insert(0,str(value)); entry.grid(row=1,column=col); entries[key]=entry
    columns=('name','speed','gain','error','reduction','spread','work','budget')
    tree=ttk.Treeview(root,columns=columns,show='headings',height=6)
    for key,label in zip(columns,['후보','구속 km/h','변화 km/h','목표 오차 cm','오차 감소 cm','가정 산포 cm','일 증가 J','일·힘 검사']):
        tree.heading(key,text=label); tree.column(key,width=133)
    tree.pack(fill='x',padx=10)
    canvas=tk.Canvas(root,width=1120,height=365,bg='white'); canvas.pack(pady=10)
    detail=tk.StringVar(); ttk.Label(root,textvariable=detail,wraplength=1110).pack()
    ttk.Label(root,text='회색: 기준 / 파랑: 선택 후보. 전면 좌표는 투수에서 본 +x. 양력 변위는 같은 이탈 상태의 무양력 궤적 대비입니다.').pack(pady=8)
    current={}
    def draw(event=None):
        result=current.get('result')
        if not result:return
        selected=tree.selection(); index=int(selected[0]) if selected else 0
        row=result['records'][index]; base=result['records'][0]; canvas.delete('all')
        if row['status']!='ok':
            detail.set('비행 제외: '+row.get('reason')+' | 원시 모델은 약 21.7도 하향입니다. 공통 방향 가정은 원인 진단용이며 실제 교정이 아닙니다.')
            canvas.create_text(560,160,text='현재 이탈 상태로는 홈 플레이트 이전에 지면에 도달합니다.\n실제 공 이탈 방향 검증이 필요합니다.',font=('Malgun Gothic',18))
            return
        canvas.create_text(280,18,text='측면 궤적 (m)');canvas.create_text(840,18,text='홈 플레이트 도착점과 가정 반복 편차')
        # Shared scale for baseline/candidate; include even multi-metre misses.
        side=lambda p:(35+p[1]*29,325-p[2]*110)
        cloud=[result['assumptions']['target_xz_m']]
        for item in (base,row):
            if item['status']=='ok':cloud.extend(item['endpoints_xz_m']);cloud.append(item['plate_xz_m'])
        cloud=np.array(cloud);lo=cloud.min(0)-.15;hi=cloud.max(0)+.15
        front=lambda p:(590+(p[0]-lo[0])/(hi[0]-lo[0])*490,320-(p[1]-lo[1])/(hi[1]-lo[1])*270)
        canvas.create_text(840,345,text=f'좌우 {lo[0]:.2f}~{hi[0]:.2f}m / 높이 {lo[1]:.2f}~{hi[1]:.2f}m (공통 축척)')
        for item,color in [(base,'#999999'),(row,'#1267bd')]:
            if item['status']!='ok':continue
            coords=[c for point in item['path_m'] for c in side(point)]
            canvas.create_line(*coords,fill=color,width=2)
            for point in item['endpoints_xz_m']:
                x,y=front(point);canvas.create_oval(x-2,y-2,x+2,y+2,outline=color)
            x,y=front(item['plate_xz_m']);canvas.create_oval(x-5,y-5,x+5,y+5,fill=color)
        x,y=front(result['assumptions']['target_xz_m']);canvas.create_line(x-8,y,x+8,y,fill='red',width=2);canvas.create_line(x,y-8,x,y+8,fill='red',width=2)
        detail.set(f"선택 {row['name']} | 입력 수정 {row['joint_parameters']}\n실제 적분의 관절 차이(도): {row['max_joint_rotation_deg']} | 공 위치 최대 차이 {row['max_ball_shift_mm']:.3f}mm\n양력 변위 좌우/상하(cm): {row['lift_displacement_cm']} | 반복 가정 실패 {row['perturbation_failed']} | 사람 적용 승인: 없음")
    def run():
        try:
            values={k:float(e.get()) for k,e in entries.items()}
            result=evaluate(data['states'],target=(values['x'],values['z']),cl=values['cl'],cd=values['cd'],jitter_deg=values['jitter'],elevation_offset_deg=values['offset'])
            result['provenance']={k:v for k,v in data.items() if k!='states'}
            (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            current['result']=result
            for item in tree.get_children():tree.delete(item)
            for i,row in enumerate(result['records']):
                if row['status']=='ok':
                    numbers=[row[k] for k in ['release_speed_kmh','speed_change_kmh','target_error_cm','target_error_reduction_cm','perturbation_spread_cm','work_change_j']]
                    vals=[row['name']]+[f'{n:.3f}' if n is not None else 'N/A' for n in numbers]+['통과(표본 검사)' if row['sampled_budget_pass'] else '초과']
                else:vals=[row['name'],f"{row['release_speed_kmh']:.3f}",'—','비행 제외','—','—',f"{row['work_change_j']:.3f}",'통과(표본 검사)' if row['sampled_budget_pass'] else '초과']
                tree.insert('', 'end',iid=str(i),values=vals)
            tree.selection_set('0');draw()
        except (ValueError,KeyError) as error:messagebox.showerror('입력 확인',str(error))
    ttk.Button(controls,text='가정 적용 / 결과 저장',command=run).grid(row=1,column=6,padx=12)
    tree.bind('<<TreeviewSelect>>',draw);run();root.mainloop()
