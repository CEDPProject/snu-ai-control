"""관리 프로세스 인계가 기존 worker를 재실행하지 않는지 실제 subprocess로 검사."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'src/scripts/control/savings_resource_supervisor.py'
spec = importlib.util.spec_from_file_location('resource_supervisor', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FAKE_BASE = '''
import json, os, sys, time, hashlib, fcntl, subprocess
from pathlib import Path
def write_json(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value)); tmp.replace(path)
def read_status(directory):
    path=Path(directory)/'status.json'
    return json.loads(path.read_text()) if path.exists() else {'status':'pending'}
def validate(root, manifest): pass
def now(): return str(time.time())
def available_memory_gb(): return 100.
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__ == '__main__':
    root=Path(sys.argv[sys.argv.index('--root')+1])
    lock=(root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    worker=subprocess.Popen([sys.executable,
        str(root/'source_snapshot/src/scripts/control/full_control_matrix.py'),
        '--worker-config',str(root/'case0/configs.yaml'),'--method','fake'])
    m=json.loads((root/'manifest.json').read_text())
    m.update(supervisor_pid=os.getpid(),active_pids=[worker.pid])
    write_json(root/'manifest.json',m)
    worker.wait()
    time.sleep(60)
'''

FAKE_WORKER = '''
import json,os,sys,time
from pathlib import Path
d=Path(sys.argv[sys.argv.index('--worker-config')+1]).parent
with (d/'starts.txt').open('a') as f: f.write(str(os.getpid())+'\\\\n')
p=d/'status.json'
p.write_text(json.dumps({'status':'running','pid':os.getpid()}))
time.sleep(8)
p.write_text(json.dumps({'status':'completed','pid':os.getpid(),
    'affinity':sorted(os.sched_getaffinity(0)),
    'nice':os.getpriority(os.PRIO_PROCESS,0)}))
'''


class SavingsResourceSupervisorTest(unittest.TestCase):
    def test_process_identity(self):
        self.assertTrue(module.alive(os.getpid()))
        self.assertFalse(module.alive(os.getpid(), -1))
        self.assertFalse(module.worker_matches(os.getpid(), Path('/nonexistent/configs.yaml')))
        self.assertFalse(module.supervisor_matches(os.getpid(), Path('/nonexistent')))

    def test_takeover_preserves_worker_pid_and_limits_new_worker(self):
        self.exercise_takeover(expansion=False)

    def test_expansion_takeover_preserves_workers_and_runs_frozen_final_analysis(self):
        self.exercise_takeover(expansion=True)

    def exercise_takeover(self, expansion):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root/'source_snapshot/src/scripts/control'
            entry.mkdir(parents=True)
            (entry/'savings_improvement.py').write_text(FAKE_BASE)
            if expansion:
                (entry/'savings_expansion.py').write_text(FAKE_BASE)
            (entry/'full_control_matrix.py').write_text(FAKE_WORKER)
            (root/'savings_improvement_analysis.py').write_text('print("analysis OK")')
            cases = []
            for i in range(2):
                d = root/f'case{i}'
                d.mkdir()
                (d/'configs.yaml').write_text('{}')
                (d/'scope_audit.json').write_text('{}')
                cases.append({'relative_directory': f'case{i}', 'method': 'fake'})
            manifest = {'cases': cases, 'input_sources': [], 'workers': 1}
            if expansion:
                final = root/'savings_improvement_final_analysis.py'
                final.write_text(
                    'from pathlib import Path\nimport sys\n'
                    '(Path(sys.argv[1])/"final_ran.txt").write_text("frozen final analysis completed")\n')
                manifest.update(experiment_kind='m3_full_scope_expansion',
                                additional_frozen_files={final.name: module.load_base(root).digest(final)})
            (root/'manifest.json').write_text(json.dumps(manifest))
            original_configs = {f'case{i}': (root/f'case{i}/configs.yaml').read_bytes() for i in range(2)}
            entrypoint = 'savings_expansion.py' if expansion else 'savings_improvement.py'
            old = subprocess.Popen([sys.executable, str(entry/entrypoint), '--root', str(root)])
            new = None
            adopted_pid = None
            try:
                deadline = time.monotonic()+10
                while not (root/'case0/status.json').exists():
                    if time.monotonic() > deadline:
                        self.fail('Fake worker did not start')
                    time.sleep(.05)
                adopted_pid = json.loads((root/'case0/status.json').read_text())['pid']
                original_start_ticks = module.process_info(adopted_pid)['start_ticks']
                cpus = sorted(os.sched_getaffinity(0))[:2]
                with (root/'scheduler.log').open('w') as out:
                    new = subprocess.Popen([
                        sys.executable, str(SCRIPT), '--root', str(root),
                        '--workers', '2', '--cpus', ','.join(map(str, cpus)),
                        '--takeover-pid', str(old.pid),
                    ], stdout=out, stderr=subprocess.STDOUT)
                    try:
                        code = new.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        self.fail((root/'scheduler.log').read_text())
                self.assertEqual(code, 0, (root/'scheduler.log').read_text())
                old.wait(timeout=5)
                self.assertEqual(old.returncode, -signal.SIGKILL)
                state = json.loads((root/'case0/status.json').read_text())
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['pid'], adopted_pid)
                self.assertEqual(state['affinity'], cpus)
                self.assertEqual(state['nice'], 19)
                other = json.loads((root/'case1/status.json').read_text())
                self.assertEqual(other['affinity'], cpus)
                self.assertEqual(other['nice'], 19)
                self.assertEqual((root/'case0/starts.txt').read_text().count(str(adopted_pid)), 1)
                self.assertEqual(len(list(root.glob('case0_attempt*'))), 0)
                result = json.loads((root/'manifest.json').read_text())
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['counts']['completed'], 2)
                limits = json.loads((root/'resource_limits.json').read_text())
                self.assertEqual(limits['logical_cpu_ids'], cpus)
                self.assertEqual(limits['workers'], 2)
                self.assertEqual(result['workers'], 2)
                self.assertEqual(result['initial_resource_configuration']['workers'], 1)
                handoff = json.loads(next(root.glob('supervisor_handoff_*.json')).read_text())
                self.assertEqual(handoff['adopted_worker_identity'][str(adopted_pid)]['start_ticks'],
                                 original_start_ticks)
                for key, value in original_configs.items():
                    self.assertEqual((root/key/'configs.yaml').read_bytes(), value)
                if expansion:
                    self.assertEqual(result['final_analysis_exit_code'], 0)
                    self.assertEqual((root/'final_ran.txt').read_text(), 'frozen final analysis completed')
            finally:
                for proc in (new, old):
                    if proc and proc.poll() is None:
                        proc.kill()
                        proc.wait()
                if adopted_pid and module.alive(adopted_pid):
                    os.kill(adopted_pid, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
