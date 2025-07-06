# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import subprocess

if __name__ == "__main__":
    kernels = [
        'kernel_kernel'
        for i in range(5)
    ]
    scripts_path = [
        # TODO: The first two kernels are not autotuned in the script.
        'util/cumsum.py',
        'common/chunk_scaled_dot_kkt.py',
        'gated_delta_rule/wy_fast.py',
        'common/chunk_o.py',
        'common/chunk_delta_h_fuse.py',
    ]
    for kernel, script_path in zip(kernels, scripts_path):
        cmd1 = [
            'ncu',
            '-k', f'regex:{kernel}',
            '--force-overwrite',
            '--target-processes', 'application-only',
            '--launch-count', '1',
            # '--launch-skip', '96',
            '--set', 'full',
            '--cache-control', 'all',
            '--clock-control', 'base',
            '--apply-rules', 'yes',
            '--import-source', 'yes',
            '--check-exit-code', 'yes',
            '--export', f'log/tilelang_{script_path.split("/")[-1].split(".")[0]}.ncu-rep',
            'python', script_path
        ]
        
        print(f"Running: {' '.join(cmd1)}")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        print(result1.stdout)
        print(result1.stderr)
