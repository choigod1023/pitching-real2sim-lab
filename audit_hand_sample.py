"""Read-only inventory of the downloaded electrogoniometry sample.

Run with the bundled document Python (openpyxl/numpy). Does not infer pitch IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import openpyxl


def audit(path):
    path = Path(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for sheet in workbook:
        iterator = sheet.iter_rows(values_only=True)
        headers = list(next(iterator))
        rows = list(iterator)
        item = dict(sheet=sheet.title, headers=headers, rows=len(rows))
        if headers == ['time(s)', 'Angle']:
            x = np.asarray(rows, dtype=float)
            dt = np.diff(x[:, 0])
            item.update(finite=bool(np.isfinite(x).all()),
                        nonincreasing_steps=int(np.sum(dt <= 0)),
                        median_dt_s=float(np.median(dt)),
                        inferred_rate_hz=float(1/np.median(dt)),
                        time_range_s=[float(x[0, 0]), float(x[-1, 0])])
        sheets.append(item)
    workbook.close()
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                md5=hashlib.md5(path.read_bytes()).hexdigest(), sheets=sheets,
                scope='Inventory only. Unlabelled sample is not assigned to baseball trials.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.path)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
