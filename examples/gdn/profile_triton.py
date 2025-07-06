# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import subprocess, os

if __name__ == "__main__":
    kernels = [
        'chunk_local_cumsum',
        'chunk_scaled_dot_kkt_fwd_kernel',
        'recompute_w_u_fwd_kernel',
        'chunk_gated_delta_rule_fwd_kernel_h_blockdim64',
        'chunk_fwd_kernel_o',
    ]
    scripts_path = [
        # TODO: This profile script is not triton-autotuned.
        # You can first autotune the kernels in FLA and then set the best config in your function call.
        'util/cumsum.py',
        'common/chunk_scaled_dot_kkt.py',
        'gated_delta_rule/wy_fast.py',
        'common/chunk_o.py',
        'common/chunk_delta_h.py',
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
            '--export', f'log/triton_{script_path.split("/")[-1].split(".")[0]}.ncu-rep',
            'python', script_path
        ]
        
        print(f"Running: TRITON_PRINT_AUTOTUNING=1 {' '.join(cmd1)}")
        # result1 = subprocess.run(cmd1, capture_output=True, text=True)
        env = os.environ.copy()
        env['TRITON_PRINT_AUTOTUNING'] = '1'
        result1 = subprocess.run(cmd1, capture_output=True, text=True, env=env)
        print(result1.stdout)
        print(result1.stderr)
