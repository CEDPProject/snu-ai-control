"""원본 전체 비교 run의 source_snapshot에 지정 파일을 반영하고 manifest 해시·이력을 갱신한다.

`--resume`은 snapshot 해시를 검증하므로 snapshot을 손으로 고치면 재개가 거부된다. 이 도구는
(1) supervisor/worker가 모두 종료됐는지 확인하고, (2) 현재 snapshot이 manifest 해시와 일치하는지
검증한 뒤, (3) live `src`의 지정 파일을 복사하고, (4) `manifest.snapshot_revisions`에
파일별 이전/이후 해시와 사유를 남긴다. 기본 파일 묶음은 HiGHS hang watchdog이다.

사용:
  python src/scripts/control/apply_snapshot_revision.py --root <run root> [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from runners.full_control_benchmark_runner import digest, write_json  # noqa: E402

WATCHDOG_FILES = [
    'controllers/guarded_milp.py',
    'controllers/mixed_integer_mpc.py',
    'controllers/control_benchmark.py',
    'runners/full_control_benchmark_runner.py',
]
REASON = ('HiGHS hang watchdog (MPC_SOLVER_HARD_TIMEOUT_SECONDS); results change only for solves '
          'exceeding the hard limit, which never completed in the original run')


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--files', nargs='*', default=WATCHDOG_FILES)
    parser.add_argument('--reason', default=REASON)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    snapshot = root/'source_snapshot'/'src'
    running = []
    for case in manifest['cases']:
        status_path = root/case['relative_directory']/'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('status') == 'running' and alive(status.get('pid')):
                running.append(case['relative_directory'])
    if alive(manifest.get('supervisor_pid')) or running:
        raise SystemExit(f'Refusing: supervisor or workers alive ({running}). Stop them first.')
    for relative, expected in manifest['source_sha256'].items():
        if digest(snapshot/relative) != expected:
            raise SystemExit(f'Snapshot already differs from manifest: {relative}')
    changes = {}
    for relative in args.files:
        source, destination = SRC/relative, snapshot/relative
        old = digest(destination) if destination.exists() else None
        new = digest(source)
        if old != new:
            changes[relative] = {'old_sha256': old, 'new_sha256': new}
    print(json.dumps(changes, indent=2))
    if args.dry_run or not changes:
        print('DRY RUN' if args.dry_run else 'NO CHANGES')
        return 0
    for relative in changes:
        (snapshot/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC/relative, snapshot/relative)
        manifest['source_sha256'][relative] = changes[relative]['new_sha256']
    manifest.setdefault('snapshot_revisions', []).append({
        'at': datetime.now().astimezone().isoformat(), 'reason': args.reason, 'files': changes,
    })
    write_json(root/'manifest.json', manifest)
    print(f'APPLIED {len(changes)} files to {snapshot}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
