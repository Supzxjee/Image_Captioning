"""Run the exact SCST test file and retain its actual failure in notebook errors."""
from pathlib import Path
import subprocess
import sys


def run_preflight(repo, log_path='/tmp/scst_preflight.log', pattern='test_scst.py'):
    repo = Path(repo).resolve()
    args = [sys.executable, '-u', '-m', 'unittest', 'discover', '-s', str(repo/'tests'),
            '-p', pattern, '-v']
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-anchor', action='store_true')
    args = parser.parse_args()
    run_preflight(Path(__file__).resolve().parent)
    if args.include_anchor:
        run_preflight(Path(__file__).resolve().parent, '/tmp/scst_anchor_preflight.log', 'test_scst_anchor.py')
