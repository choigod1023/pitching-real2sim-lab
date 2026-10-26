"""Manual desktop smoke check; requires tkinter/display and generated states."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch
from pitch_outcomes_gui import launch


def check(root):
    controls=root.winfo_children()[2]
    entries=[w for w in controls.winfo_children() if isinstance(w,ttk.Entry)]
    button=next(w for w in controls.winfo_children() if isinstance(w,ttk.Button))
    entries[-1].delete(0,'end');entries[-1].insert(0,'20');button.invoke()
    root.update()
    canvas=next(w for w in root.winfo_children() if isinstance(w,tk.Canvas))
    assert len(canvas.find_all())>50
    tree=next(w for w in root.winfo_children() if isinstance(w,ttk.Treeview))
    tree.selection_set('4');tree.event_generate('<<TreeviewSelect>>');root.update()
    assert len(canvas.find_all())>50
    print('GUI: baseline/candidate overlay, selection and scenario controls passed')
    entries[-1].delete(0,'end');entries[-1].insert(0,'0');button.invoke()
    root.update();assert len(canvas.find_all())==1
    print('GUI: invalid raw release blocked; default results restored')
    root.destroy()


if __name__=='__main__':
    out=Path('artifacts/outcome-simulator')
    with patch.object(tk.Tk,'mainloop',check):
        launch(json.loads((out/'states.json').read_text()),out)
