import subprocess
import time
import os
import signal
import sys
import psutil
import pandas as pd

def run_process_with_timeout(cmd, timeout=12000):
    """Run a process and kill it after timeout seconds"""
    process = subprocess.Popen(cmd)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return False
    return True

def kill_previous_processes():
    """Kill any existing Python processes running the target script"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
                cmdline = proc.info['cmdline']
                if cmdline and 'autotune_per_process.py' in cmdline[0]:
                    os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def main(is_static=True, is_splitk=True, is_tma=True):
    # Script to run
    if is_splitk:
        script_path = "testing/python/dynamic/test_tilelang_dynamic_splitk_autotune.py"
    else:
        script_path = "testing/python/dynamic/test_tilelang_dynamic_autotune.py"

    current_time = time.strftime("%m%d", time.localtime(time.time()))
    kernel_save_dir = os.path.join("testing/python/dynamic/.log", f"{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}{'_tma' if is_tma else '_no_tma'}_{current_time}")
    config_file_path = "/home/tzj/workspace/dynamic/config/matmul_config_test.csv"
    
    # Command line arguments for the script
    base_args = [
        "--autotune=True"
    ]
    if is_static:
        base_args.append("--static=True")
    if is_tma:
        base_args.append("--tma=True")
    
    # Different configurations to try
    configs = []
    config_df = pd.read_csv(config_file_path)
    for index, row in config_df.iterrows():
        existing_file_path = os.path.join(kernel_save_dir, f"best_config_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}_{row['M']}_{row['N']}_{row['K']}.csv")
        # Filter out configs with M = 1 (should be moved to GEMV)
        if not os.path.exists(existing_file_path) and row["M"] > 1:
            configs.append([
                "--m=" + str(row["M"]),
                "--n=" + str(row["N"]),
                "--k=" + str(row["K"])
            ])
    
    
    for config in configs:
        # Kill any existing processes
        kill_previous_processes()
        
        # Construct full command
        cmd = [sys.executable, script_path] + base_args + config
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the process with timeout
        success = run_process_with_timeout(cmd)
        
        if not success:
            print(f"Process timed out for config: {config}")
        
        # Wait a bit before starting next process
        time.sleep(3)

if __name__ == "__main__":
    main(is_static=True, is_splitk=True, is_tma=True)
