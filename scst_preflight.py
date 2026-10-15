"""Run the exact SCST test file and retain its actual failure in notebook errors."""
from pathlib import Path
import subprocess
import sys


def run_preflight(repo, log_path='/tmp/scst_preflight.log'):
    repo = Path(repo).resolve()
    args = [sys.executable, '-u', '-m', 'unittest', 'discover', '-s', str(repo/'tests'),
            '-p', 'test_scst.py', '-v']
    print('SCST preflight:', ' '.join(args), flush=True)
    result = subprocess.run(args, cwd=repo, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors='replace')
    Path(log_path).write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'SCST preflight failed (exit {result.returncode}). Full log: {log_path}\n'
                           + result.stdout[-12000:])
    return result.stdout


if __name__ == '__main__':
    run_preflight(Path(__file__).resolve().parent)
