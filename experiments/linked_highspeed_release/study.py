"""Audit same-row Rapsodo video links and compare conditional ball flight."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from pitch_outcomes import flight


def read_pitches(source):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relationship = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with ZipFile(source) as workbook:
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = [
            "".join(part.text or "" for part in item.findall(".//s:t", ns))
            for item in root
        ]
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        relations = ElementTree.fromstring(workbook.read("xl/worksheets/_rels/sheet1.xml.rels"))
    targets = {item.get("Id"): item.get("Target") for item in relations}
    links = {
        link.get("ref"): targets.get(link.get(relationship))
        for link in sheet.findall(".//s:hyperlink", ns)
    }
    header, pitches = {}, []
    for row in sheet.findall(".//s:sheetData/s:row", ns):
        cells = {}
        for cell in row.findall("s:c", ns):
            value = cell.findtext("s:v", "", ns)
            if cell.get("t") == "s":
                value = strings[int(value)]
            cells[cell.get("r").rstrip("0123456789")] = value
        if cells.get("A") == "Game Pitch":
            header = {label: column for column, label in cells.items()}
            continue
        if not header or not cells.get(header["Game Pitch"], "").isdigit():
            continue
        values = {key: cells.get(column, "") for key, column in header.items()}
        pitch = {
            "pitch_id": int(values["Game Pitch"]),
            "pitch_type": values["Pitch Type"],
            "video": links.get(header["HS Vid"] + row.get("r")),
        }
        if pitch["pitch_type"] != "FB":
            pitches.append(pitch)
            continue
        pitch.update({
            "speed_mph": float(values["RelSpeed (MPH)"]),
            "rpm": float(values["SpinRate"]),
            "true_rpm": float(values["TrueSpinRelease"]),
            "direction": values["Spin Direction"],
            "ivb_in": float(values["InducedVertBreak (Inches)"]),
            "hb_in": float(values["HorzBreak (Inches)"]),
            "horizontal_angle_deg": float(values["HorzRelAngle"]),
            "vertical_angle_deg": float(values["VertRelAngle"]),
            "release_height_ft": float(values["RelHeight (Feet)"]),
        })
        if values.get("PlateLocHeight (Inches)"):
            pitch["plate_height_in"] = float(values["PlateLocHeight (Inches)"])
        pitches.append(pitch)
    if not pitches or len({pitch["pitch_id"] for pitch in pitches}) != len(pitches):
        raise ValueError("Expected unique, nonempty Game Pitch identifiers")
    return pitches


def launch_basis(pitch):
    azimuth, elevation = np.deg2rad([pitch["horizontal_angle_deg"], pitch["vertical_angle_deg"]])
    forward = np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation),
    ])
    right = np.array([np.cos(azimuth), -np.sin(azimuth), 0.])
    up = np.cross(right, forward)
    return forward, right, up


def predict_movement(pitch, distance=16.8, cd=.35, dt=.001, *, spin_axis=None, active_rpm=None):
    """Return horizontal and induced vertical break in cm, and the flight path."""
    speed = pitch["speed_mph"] * .44704
    forward, right, up = launch_basis(pitch)
    if spin_axis is None:
        hours, minutes = map(int, pitch["direction"].split(":"))
        if not (0 <= hours <= 12 and 0 <= minutes < 60):
            raise ValueError("Expected a Rapsodo clock direction")
        tilt = np.deg2rad(30 * (hours % 12) + .5 * minutes)
        spin_axis = np.cross(forward, np.cos(tilt) * up + np.sin(tilt) * right)
    spin_parameter = .0366 * (2 * np.pi * (pitch["true_rpm"] if active_rpm is None else active_rpm) / 60) / speed
    cl = 1.5 * spin_parameter if spin_parameter <= .1 else .09 + .6 * spin_parameter
    position = [0., 0., pitch["release_height_ft"] * .3048]
    path, _ = flight(position, speed * forward, spin_axis, cl, cd, distance, dt)
    baseline, _ = flight(position, speed * forward, spin_axis, 0., cd, distance, dt)
    movement = 100 * (path[-1, [0, 2]] - baseline[-1, [0, 2]])
    return movement, path[:, :3]


def audit(source, optical_rpm=None):
    pitches = read_pitches(source)
    optical_rpm = optical_rpm or {}
    results = []
    for pitch in pitches:
        if pitch["pitch_type"] != "FB":
            continue
        movement, path = predict_movement(pitch)
        fine_movement, _ = predict_movement(pitch, dt=.0005)
        sensitivity, failed = [], []
        for distance in (16.2, 17.4):
            for cd in (.30, .40):
                try:
                    sensitivity.append(predict_movement(pitch, distance, cd)[0])
                except ValueError as exc:
                    failed.append({"distance_m": distance, "cd": cd, "error": str(exc)})
        sensitivity = np.asarray(sensitivity)
        reference = 2.54 * np.array([pitch["hb_in"], pitch["ivb_in"]])
        row = {
            **pitch,
            "speed_kmh": pitch["speed_mph"] * 1.609344,
            "recorded_hb_ivb_cm": reference.tolist(),
            "conditional_hb_ivb_cm": movement.tolist(),
            "conditional_minus_recorded_cm": (movement - reference).tolist(),
            "integration_step_difference_cm": np.abs(movement - fine_movement).tolist(),
            "sensitivity_min_hb_ivb_cm": sensitivity.min(axis=0).tolist() if sensitivity.size else None,
            "sensitivity_max_hb_ivb_cm": sensitivity.max(axis=0).tolist() if sensitivity.size else None,
            "sensitivity_exclusions": failed,
            "path_m": path.tolist(),
        }
        if str(pitch["pitch_id"]) in optical_rpm:
            rpm = float(optical_rpm[str(pitch["pitch_id"])])
            if not pitch["video"] or not np.isfinite(rpm) or rpm <= 0:
                raise ValueError("Optical RPM requires a linked video and a positive RPM")
            observed = {**pitch, "true_rpm": rpm * pitch["true_rpm"] / pitch["rpm"]}
            optical_movement, _ = predict_movement(observed)
            row.update(
                optical_rpm=rpm,
                optical_minus_recorded_rpm=rpm - pitch["rpm"],
                optical_input_hb_ivb_cm=optical_movement.tolist(),
                optical_input_minus_recorded_cm=(optical_movement - reference).tolist(),
            )
        results.append(row)
    linked = [row for row in results if row["video"]]
    pairs = [
        {
            "pitch_ids": [a["pitch_id"], b["pitch_id"]],
            "speed_difference_kmh": abs(a["speed_kmh"] - b["speed_kmh"]),
            "meets_speed_threshold": abs(a["speed_kmh"] - b["speed_kmh"]) <= 1.,
        }
        for index, a in enumerate(linked)
        for b in linked[index + 1:]
    ]
    return {
        "workbook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "all_pitch_count": len(pitches),
        "linked_video_count": sum(bool(pitch["video"]) for pitch in pitches),
        "linked_four_seam_pairs": pairs,
        "four_seams": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--optical-rpm", type=Path, help="JSON mapping pitch ID to optical total RPM")
    args = parser.parse_args()
    optical = json.loads(args.optical_rpm.read_text(encoding="utf-8")) if args.optical_rpm else None
    args.output.write_text(json.dumps(audit(args.workbook, optical), indent=2), encoding="utf-8")
