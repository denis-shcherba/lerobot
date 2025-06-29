import h5py
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_h5_to_final_lerobot_format(h5_file_path, output_path):
    """
    Converts a single HDF5 file into a final, fully compliant LeRobot dataset,
    including all required metadata: info.json, stats.json, tasks.jsonl,
    and episodes.jsonl.
    """
    print("--- Starting Final LeRobot Dataset Conversion ---")
    output_path = Path(output_path)
    data_dir = output_path / "data"
    meta_dir = output_path / "meta"

    # 1. Create the required directory structure
    print(f"Creating directories at {output_path}...")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 2. Process the HDF5 file and gather all data
    all_steps_data = []
    all_episodes_metadata = [] # <<< NEW: To store metadata for episodes.jsonl
    task_description = "act_point_cloud_imitation" # Define a single task description

    print(f"Opening HDF5 file at {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        episode_keys = sorted([key for key in f.keys() if key.startswith('demo_')])
        print(f"Found {len(episode_keys)} episodes.")

        for episode_idx, group_name in enumerate(episode_keys):
            episode_group = f[group_name]
            point_clouds = episode_group['points'][:]
            path_data = episode_group['path'][:]
            states, actions = path_data, path_data
            num_steps = actions.shape[0]

            # <<< NEW: Capture metadata for this specific episode
            episode_meta = {
                "episode_index": episode_idx,
                "tasks": [task_description],
                "length": num_steps
            }
            all_episodes_metadata.append(episode_meta)

            for t in range(num_steps):
                step_data = {
                    "observation.point_cloud": point_clouds[t].tolist(),
                    "observation.state": states[t],
                    "action": actions[t],
                    "episode_index": episode_idx,
                    "frame_index": t,
                    "next.done": (t == num_steps - 1),
                }
                all_steps_data.append(step_data)

    total_frames = len(all_steps_data)
    total_episodes = len(episode_keys)
    print(f"\nTotal episodes: {total_episodes}, Total frames/timesteps: {total_frames}")

    # 3. Save trajectory data to a parquet file
    df = pd.DataFrame(all_steps_data)
    parquet_path = data_dir / "episodes.parquet"
    print(f"Saving data to {parquet_path}...")
    df.to_parquet(parquet_path)
    print("Data saved successfully.")

    # 4. Create and save info.json
    print("Generating info.json metadata...")
    with h5py.File(h5_file_path, 'r') as f:
        pc_shape = f['demo_0']['points'][0].shape
        state_shape = f['demo_0']['path'][0].shape
        action_shape = f['demo_0']['path'][0].shape

    info_dict = {
        "codebase_version": "v2.0",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/episodes.parquet",
        "features": {
            "observation.point_cloud": {"dtype": "float32", "shape": list(pc_shape)},
            "observation.state": {"dtype": "float32", "shape": list(state_shape)},
            "action": {"dtype": "float32", "shape": list(action_shape)},
            "episode_index": {"dtype": "int64", "shape": []},
            "frame_index": {"dtype": "int64", "shape": []},
            "next.done": {"dtype": "bool", "shape": []},
        }
    }
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info_dict, f, indent=4)
    print("Metadata saved to meta/info.json")

    # 5. Create and save tasks.jsonl
    print("Generating tasks.jsonl metadata...")
    task_entry = {"task_index": 0, "task": task_description}
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps(task_entry) + '\n')
    print("Task metadata saved to meta/tasks.jsonl")

    # 6. <<< NEW: Create and save the episodes.jsonl file
    print("Generating episodes.jsonl metadata...")
    episodes_jsonl_path = meta_dir / "episodes.jsonl"
    with open(episodes_jsonl_path, 'w') as f:
        for episode_meta in all_episodes_metadata:
            f.write(json.dumps(episode_meta) + '\n')
    print(f"Episode metadata saved to {episodes_jsonl_path}")

    # 7. Calculate and save statistics in stats.json
    print("Calculating dataset statistics for stats.json...")
    stats_dict = {}
    features_to_normalize = ["observation.point_cloud", "observation.state", "action"]
    for feature in features_to_normalize:
        all_data = np.array(df[feature].tolist())
        if all_data.ndim == 3:
            all_data = all_data.reshape(-1, all_data.shape[-1])
        stats_dict[feature] = {
            "mean": all_data.mean(axis=0).tolist(),
            "std": all_data.std(axis=0).tolist(),
            "min": all_data.min(axis=0).tolist(),
            "max": all_data.max(axis=0).tolist(),
        }
    print(f"  - Calculated stats for '{feature}'")

    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print("Statistics saved to meta/stats.json")

    print("\n--- LeRobot Dataset creation complete! ---")
    print(f"Your dataset is ready at: {output_path}")

if __name__ == "__main__":
    my_data_file = "./demos/demo_test_deltas.h5"
    local_save_path = "./lerobot_datasets/my_final_lerobot_dataset"

    # Run the complete conversion
    convert_h5_to_final_lerobot_format(my_data_file, local_save_path)