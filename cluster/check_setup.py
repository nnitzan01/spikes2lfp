"""
Pre-flight check for the cluster: software, GPU code path and data, in about a minute.

Run on a GPU node from the repository root, e.g.
    srun --gres=gpu:a100:1 --mem=16G --time=00:15:00 --pty bash
    source cluster/env.sh && python cluster/check_setup.py

Every line starts with OK or FAIL; fix any FAIL before submitting the array job.
"""
import importlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

failures = []


def report(ok, message):
    print(('OK   ' if ok else 'FAIL ') + message, flush=True)
    if not ok:
        failures.append(message)


def check_imports():
    for name in ('numpy', 'scipy', 'pandas', 'sklearn', 'torch', 'pynwb', 'allensdk'):
        try:
            module = importlib.import_module(name)
            report(True, f'import {name} {getattr(module, "__version__", "")}')
        except Exception as e:                       # noqa: BLE001 - report any import problem
            report(False, f'import {name}: {e}')


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        report(False, 'CUDA not available (are you on a GPU node, with --gres=gpu:1?)')
        return
    try:
        cudnn = torch.backends.cudnn.version()
    except RuntimeError as e:                       # e.g. an incompatible cuDNN on LD_LIBRARY_PATH
        report(False, f'cuDNN: {str(e).splitlines()[0]}')
        return
    report(True, f'GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}, '
                 f'cuDNN {cudnn}')

    # the exact GPU code path of the pipeline: causal CNN-LSTM training step and SequenceIG
    # (cuDNN RNN backward with the recurrent layer in train mode and the rest in eval mode)
    import config
    import models
    from integrated_gradients import SequenceIG
    torch.manual_seed(0)
    device = torch.device('cuda')
    model = models.build_lstm(300, device)
    x = (torch.rand(8, config.SEQLENGTH, 300, device=device) < 0.02).float()
    model.model.train()
    out, _ = model.model(x)
    out.mean().backward()
    report(True, f'CNN-LSTM forward/backward on GPU ({model.n_parameters()} parameters for 300 units)')

    t0 = time.time()
    attr, f_x, f_b, delta = SequenceIG(model.model, n_steps=config.IG_N_STEPS).attribute(x)
    seconds = (time.time() - t0) / len(x)
    rel = (delta.abs() / (f_x - f_b).abs().clamp_min(1e-8)).max().item()
    report(rel < 1e-2, f'SequenceIG on GPU: {seconds:.3f} s per window, max relative completeness error {rel:.1e}')


def check_data():
    import os
    import paths
    root = Path(os.environ.get('SPIKES2LFP_DATA_ROOT', ''))
    report(root.is_dir(), f'SPIKES2LFP_DATA_ROOT = {root}')
    if not root.is_dir():
        return
    report((root / 'visual-behavior-neuropixels-0.4.0' / 'project_metadata' / 'probes.csv').exists(),
           'project_metadata/probes.csv present')
    sessions = (REPO / 'cluster' / 'sessions.txt').read_text().split()
    session_root = root / 'visual-behavior-neuropixels-0.4.0' / 'behavior_ecephys_sessions'
    missing = [s for s in sessions if not (session_root / s).is_dir()]
    report(not missing, f'{len(sessions) - len(missing)}/{len(sessions)} session directories present'
                        + (f' (missing: {", ".join(missing[:5])}{"..." if len(missing) > 5 else ""})' if missing else ''))
    no_lfp = [s for s in sessions if (session_root / s).is_dir() and not list((session_root / s).glob('probe_*_lfp.nwb'))]
    report(not no_lfp, 'LFP NWB files present for all session directories'
                       + (f' (none for: {", ".join(no_lfp[:5])})' if no_lfp else ''))
    report((REPO / 'tables' / 'units_info.csv').exists(), 'tables/units_info.csv present')
    out = root / paths.OUTPUT_SUBDIR
    try:
        out.mkdir(exist_ok=True)
        (out / '.write_test').write_text('ok')
        (out / '.write_test').unlink()
        report(True, f'output directory {out} is writable')
    except OSError as e:
        report(False, f'cannot write to {out}: {e}')


if __name__ == '__main__':
    for check in (check_imports, check_gpu, check_data):
        try:
            check()
        except Exception as e:                       # noqa: BLE001 - report and keep checking
            report(False, f'{check.__name__}: {type(e).__name__}: {e}')
    print(f'\n{"All checks passed." if not failures else f"{len(failures)} check(s) failed."}')
    sys.exit(1 if failures else 0)
