import os
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

def run_backtest():
    project_root = os.getcwd()
    signal_path = os.path.join(project_root, "data", "signal.parquet")
    output_dir = os.path.join(project_root, "data", "weights")
    logs_dir = os.path.join(project_root, "logs")

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Define Slurm Configuration
    slurm_config = SlurmConfig(
        n_cpus=8,
        mem="32G",
        time="03:00:00",
        mail_type="BEGIN,END,FAIL",
        max_concurrent_jobs=30,
    )

    # Define Backtest Configuration
    config = BacktestConfig(
        signal_name="my_first_signal",  # Name your signal
        data_path=signal_path,
        gamma=10,                       # Update Risk aversion
        project_root=project_root,
        byu_email="user@byu.edu",       # Update this
        constraints=[],                 # Add constraints
        slurm=slurm_config,
        output_dir=output_dir,
        logs_dir=logs_dir
    )

    runner = BacktestRunner(config)
    runner.submit(dry_run=False)

if __name__ == "__main__":
    run_backtest()
