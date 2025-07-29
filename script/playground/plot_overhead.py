import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse

def read_data(folder_path, model_name):
    pattern = f"{folder_path}/*_{model_name}.json"
    data = []
    for file_name in glob.glob(pattern):
        with open(file_name, 'r') as file:
            json_data = json.load(file)
            data.append(json_data)
    return data

def process_data(data):
    df = pd.DataFrame(data)
    # df['total_snap_cnt'] = df['total_snap_cnt'].astype(str)
    return df.groupby(['snap_policy', 'total_snap_cnt']).agg(
        total_iter_time=('total_iter_time', 'mean'),
        total_snap_time=('total_snap_time', 'mean'),
        snap_overhead=('snap_overhead', 'mean')
    ).reset_index()

def plot_stacked_bar(df, model_name):
    policies = df['snap_policy'].unique()
    offsets = dict(zip(policies, range(len(policies))))
    bar_width = 0.2

    fig, ax = plt.subplots()
    for policy in policies:
        policy_df = df[df['snap_policy'] == policy]
        offsets = len(policies) * bar_width * (policy_df['snap_policy'].map(policies.tolist().index))
        ax.bar(policy_df['total_snap_cnt'] + offsets, policy_df['total_iter_time'], label='Iteration Time', width=bar_width, align='center')
        ax.bar(policy_df['total_snap_cnt'] + offsets, policy_df['total_snap_time'], bottom=policy_df['total_iter_time'], label='Snapshot Time', width=bar_width, align='center')

    ax.set_xlabel('Total Snapshot Count')
    ax.set_ylabel('Time')
    ax.set_title(f'Time Breakdown by Snapshot Policy and Count for {model_name}')
    ax.set_xticks(df['total_snap_cnt'].unique())
    ax.set_xticklabels(df['total_snap_cnt'].unique())
    ax.legend()
    plt.savefig(f"{model_name}_time.pdf")

def plot_snap_overhead_bar(df, model_name):
    policies = df['snap_policy'].unique()
    bar_width = 0.2
    fig, ax = plt.subplots()

    for index, policy in enumerate(policies):
        policy_df = df[df['snap_policy'] == policy]
        ax.bar(policy_df['total_snap_cnt'] + (index * bar_width), policy_df['snap_overhead'], width=bar_width, label=policy)

    ax.set_xlabel('Total Snapshot Count')
    ax.set_ylabel('Snapshot Overhead')
    ax.set_title(f'Snapshot Overhead by Policy and Count for {model_name}')
    ax.set_xticks(df['total_snap_cnt'].unique())
    ax.set_xticklabels(df['total_snap_cnt'].unique())
    ax.legend()
    plt.savefig(f"{model_name}_overhead.pdf")

def main(folder_path, model_name):
    data = read_data(folder_path, model_name)
    processed_data = process_data(data)
    plot_stacked_bar(processed_data, model_name)
    plot_snap_overhead_bar(processed_data, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting performance metrics for models.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the data files.")
    parser.add_argument("model_name", type=str, help="Name of the model to plot data for.")
    
    args = parser.parse_args()
    
    main(args.folder_path, args.model_name)
