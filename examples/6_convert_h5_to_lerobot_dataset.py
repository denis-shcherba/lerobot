import h5py
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_h5_to_final_lerobot_format(h5_file_path, output_path):
    """
    Converts a single HDF5 file into a final, fully compliant LeRobot dataset.
    This version reshapes the image statistics to (C, 1, 1) to fix library broadcasting issues.
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
    all_episodes_metadata = []
    observation_key = None
    observation_type = None
    task_description = None

    print(f"Opening HDF5 file at {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        episode_keys = sorted([key for key in f.keys() if key.startswith('demo_')])
        if not episode_keys:
            raise ValueError("No episodes found in HDF5 file.")
        print(f"Found {len(episode_keys)} episodes.")

        # --- Auto-detect observation type from the first episode ---
        first_episode_group = f[episode_keys[0]]
        if 'points' in first_episode_group:
            observation_key = 'points'
            observation_type = 'point_cloud'
            task_description = "act_point_cloud_imitation"
            print("Detected 'point_cloud' data.")
        elif 'rgb' in first_episode_group:
            observation_key = 'rgb'
            observation_type = 'image'
            task_description = "act_rgb_imitation"
            print("Detected 'image' data under the key 'rgb'.")
        else:
            raise ValueError("Could not find 'points' or 'rgb' data in the first episode.")

        # --- Process all episodes ---
        for episode_idx, group_name in enumerate(episode_keys):
            episode_group = f[group_name]
            observation_data = episode_group[observation_key][:]
            path_data = episode_group['path'][:]
            states, actions = path_data, path_data
            num_steps = actions.shape[0]

            episode_meta = {
                "episode_index": episode_idx,
                "tasks": [task_description],
                "length": num_steps
            }
            all_episodes_metadata.append(episode_meta)

            for t in range(num_steps):
                step_obs_data = observation_data[t]
                
                # Transpose image data from (H, W, C) to (C, H, W)
                if observation_type == 'image':
                    step_obs_data = step_obs_data.transpose(2, 0, 1)

                step_data = {
                    f"observation.{observation_type}": step_obs_data.tolist(),
                    "observation.state": states[t].tolist(),
                    "action": actions[t].tolist(),
                    "task_index": [0],
                    "episode_index": [episode_idx],
                    "frame_index": t,
                    "timestamp": [float(t)],
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

    # 4. Create and save info.json dynamically
    print("Generating info.json metadata...")
    with h5py.File(h5_file_path, 'r') as f:
        obs_data_sample = f['demo_0'][observation_key][0]
        
        if observation_type == 'image':
            obs_data_sample = obs_data_sample.transpose(2, 0, 1)

        obs_shape = obs_data_sample.shape
        obs_dtype = str(obs_data_sample.dtype)
        state_shape = f['demo_0']['path'][0].shape
        action_shape = f['demo_0']['path'][0].shape

    info_dict = {
        "codebase_version": "v2.0",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "chunks_size": 64,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/episodes.parquet",
        "features": {
            f"observation.{observation_type}": {
                "dtype": "image" if observation_type == 'image' else obs_dtype,
                "shape": list(obs_shape)
            },
            "observation.state": {"dtype": "float32", "shape": list(state_shape)},
            "action": {"dtype": "float32", "shape": list(action_shape)},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": []},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": []},
        }
    }
    
    if observation_type == 'image':
        image_feature_key = f"observation.{observation_type}"
        info_dict["features"][image_feature_key]["names"] = ["channel", "height", "width"]

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info_dict, f, indent=4)
    print("Metadata saved to meta/info.json")

    print("Generating tasks.jsonl metadata...")
    task_entry = {"task_index": 0, "task": task_description}
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps(task_entry) + '\n')
    print("Task metadata saved to meta/tasks.jsonl")

    print("Generating episodes.jsonl metadata...")
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for episode_meta in all_episodes_metadata:
            f.write(json.dumps(episode_meta) + '\n')
    print("Episode metadata saved to meta/episodes.jsonl")

    # 7. Calculate and save statistics in stats.json
    print("Calculating dataset statistics for stats.json...")
    stats_dict = {}
    features_to_normalize = [f"observation.{observation_type}", "observation.state", "action", "timestamp"]
    
    for feature in features_to_normalize:
        all_data = np.array(df[feature].tolist(), dtype=np.float32)

        is_image = feature == f"observation.{observation_type}" and observation_type == 'image'
        
        if is_image:
            # For images (N, C, H, W), calculate stats per channel.
            num_channels = all_data.shape[1]
            all_data = all_data.transpose(0, 2, 3, 1).reshape(-1, num_channels)
            print(f"  - Calculating per-channel stats for '{feature}' (shape: {all_data.shape})")
        else:
            if all_data.ndim == 1:
                all_data = all_data.reshape(-1, 1)

        mean_vals = all_data.mean(axis=0)
        std_vals = all_data.std(axis=0)
        min_vals = all_data.min(axis=0)
        max_vals = all_data.max(axis=0)

        # ---- THE FIX: Reshape image stats to (C, 1, 1) ----
        if is_image:
            print(f"  - Reshaping stats for '{feature}' to be broadcastable.")
            mean_vals = mean_vals.reshape(-1, 1, 1)
            std_vals = std_vals.reshape(-1, 1, 1)
            min_vals = min_vals.reshape(-1, 1, 1)
            max_vals = max_vals.reshape(-1, 1, 1)
            
        stats_dict[feature] = {
            "mean": mean_vals.tolist(),
            "std": std_vals.tolist(),
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
        }
        print(f"  - Calculated stats for '{feature}'")

    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print("Statistics saved to meta/stats.json")

    print("\n--- LeRobot Dataset creation complete! ---")
    print(f"Your dataset is ready at: {output_path}")


if __name__ == "__main__":
    my_data_file = "./demos/rgbtest.h5"
    local_save_path = "./lerobot_datasets/my_final_rgb_dataset"
    print(f"About to generate dataset at: {local_save_path}")
    print("Please ensure you have deleted this directory first to avoid errors.")
    convert_h5_to_final_lerobot_format(my_data_file, local_save_path)