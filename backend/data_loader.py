import csv
from typing import List


def load_csv(path: str) -> List[str]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(', '.join([f"{k}: {v}" for k, v in row.items()]))
    return rows
