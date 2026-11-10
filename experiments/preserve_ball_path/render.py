"""Show the protected ball path and reference points on the original video."""
import argparse
from pathlib import Path

from experiments.release_conditioned_ball.render import render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","report","destination"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--font",required=True,type=Path)
    parser.add_argument("--outdoor",default="academy-000002")
    args = parser.parse_args()
    render(args.manifest,args.report,args.destination,args.font,
           cases=(("prime-22","Prime 22 · 기존 경로 보존"),
                  ("mlb-776202-low","MLB · 놓친 공 복구"),
                  ("mlb-778561-low","MLB · 릴리스 전 오검출 비교"),
                  (args.outdoor,"아카데미 · 미사용 야외 투구")),
           variant="anchored",heading="기존 경로를 유지하고 후속 영상에서 새 점을 확인",variant_label="경로 보존")
