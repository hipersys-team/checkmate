import paramiko
import threading
import argparse

username = "ankitbwj"
all_nodes = ["sr01", "sr02", "sr04", "sr05"]
conda_env_names = ["ckpt", "ckpt", "base", "ckpt"]
max_threads_per_node = 28
remote_script_path = f"/home/{username}/innet_ckpt/storage/models/run_opt_bench.py"


# Function to execute the script on a remote node
def run_script_on_node(node, command=None):
    """
    Executes a command on a given node with specified environment variables.
    :param node: Node hostname
    :param env_vars: Environment variables (string)
    :param cmd: Command to execute (string)
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=node,
            port=22,
            allow_agent=True,
            username=username,
        )

        stdin, stdout, stderr = ssh.exec_command(command)

        # Capture the output and errors
        output = stdout.read().decode()
        errors = stderr.read().decode()
        print(f"Output from {node}:\n{output}")
        if errors and "Warning" not in errors:
            print(f"Errors from {node}:\n{errors}")

        ssh.close()
    except Exception as e:
        print(f"Failed to run script on {node}: {e}")


def distribute_torchrun_tasks(max_threads, script_path, model):
    """
    Distributes torchrun tasks across nodes based on max threads available.
    :param max_threads: Total maximum threads to use across all nodes.
    :param script_path: Path to the script to execute with torchrun.
    """
    # Validate the input
    if max_threads <= 0:
        raise ValueError("max_threads must be a positive integer.")

    # Determine nodes and thread allocation
    threads_remaining = max_threads
    threads_allocations = []
    for node in all_nodes:
        if threads_remaining <= 0:
            break
        threads_for_node = min(max_threads_per_node, threads_remaining)
        threads_allocations.append((node, threads_for_node))
        threads_remaining -= threads_for_node

    if threads_remaining > 0:
        raise ValueError(
            f"Insufficient nodes to allocate {max_threads} threads. Increase the number of available nodes."
        )

    # Total nodes being used
    nnodes = len(threads_allocations)

    # Launch tasks on nodes
    threads = []
    for node_rank, (node, threads_for_node) in enumerate(threads_allocations):
        # Construct the torchrun command
        conda_env = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env_names[node_rank]}"
        env_vars = f"OMP_NUM_THREADS={threads_for_node} CUDA_VISIBLE_DEVICES=''"
        command = f"torchrun --nnodes={nnodes} --node-rank={node_rank} --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint={all_nodes[0]} {script_path} --model={model}"
        full_command = f"{conda_env} && {env_vars} {command}"
        print(f"Running on {node} with {threads_for_node} threads: {full_command}")
        thread = threading.Thread(
            target=run_script_on_node,
            args=(
                node,  # Node name
                full_command,  # Command to run
            ),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch DDP benchmark")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=[
            "resnet50",
            "resnet152",
            "vgg11",
            "vit_h_14",
            "gpt2",
            "gpt3xl",
            "gpt3_6_7B",
            "7B",
            "13B",
        ],
        type=str,
        help="model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    total_cores_needed = len(all_nodes) * max_threads_per_node

    for cores in range(0, total_cores_needed + 1, 112):
        if cores == 0:
            cores = 1
        print(f"Executed with {cores} cores.")
        distribute_torchrun_tasks(cores, remote_script_path, args.model)
        input("Press Enter to continue...; ctrl+c to exit")
