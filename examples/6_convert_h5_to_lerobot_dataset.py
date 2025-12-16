import h5py
import json
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# --- Helper Class for Streaming Statistics ---
class StreamingStats:
    def __init__(self, shape, is_image=False):
        self.n = 0
        self.shape = shape
        self.is_image = is_image
        
        # Initialize accumulators
        # If image (C, H, W), we calculate stats per channel -> shape (C,)
        # If state (D,), we calculate per dimension -> shape (D,)
        if self.is_image:
            self.stat_shape = (shape[0],) # Channels only
        else:
            self.stat_shape = shape
            
        self.sum = np.zeros(self.stat_shape, dtype=np.float64)
        self.sum_sq = np.zeros(self.stat_shape, dtype=np.float64)
        self.min = np.full(self.stat_shape, np.inf, dtype=np.float64)
        self.max = np.full(self.stat_shape, -np.inf, dtype=np.float64)

    def update(self, data):
        """
        data: single frame numpy array.
        """
        # Handle Images: Flatten spatial dims to accumulate stats per channel
        if self.is_image:
            # Data comes in as (C, H, W). We want stats over H*W for this image
            # Transpose to (H, W, C) -> Reshape to (pixels, C)
            flat_data = data.transpose(1, 2, 0).reshape(-1, self.stat_shape[0]).astype(np.float64)
        else:
            # Ensure 1D for states/actions
            flat_data = data.reshape(1, -1).astype(np.float64)

        # Update stats
        count = flat_data.shape[0]
        self.n += count
        self.sum += flat_data.sum(axis=0)
        self.sum_sq += (flat_data ** 2).sum(axis=0)
        self.min = np.minimum(self.min, flat_data.min(axis=0))
        self.max = np.maximum(self.max, flat_data.max(axis=0))

    def get_final_stats(self):
        mean = self.sum / self.n
        # Variance = E[X^2] - (E[X])^2
        variance = (self.sum_sq / self.n) - (mean ** 2)
        # Numerical stability clip
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)

        # Reshape to (C, 1, 1) for images if needed, or flat for others
        if self.is_image:
            return {
                "mean": mean.reshape(-1, 1, 1).tolist(),
                "std": std.reshape(-1, 1, 1).tolist(),
                "min": self.min.reshape(-1, 1, 1).tolist(),
                "max": self.max.reshape(-1, 1, 1).tolist()
            }
        else:
            return {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": self.min.tolist(),
                "max": self.max.tolist()
            }

def convert_h5_to_final_lerobot_format(h5_file_path, output_path):
    print("--- Starting Final LeRobot Dataset Conversion (Memory Optimized) ---")
    output_path = Path(output_path)
    data_dir = output_path / "data"
    meta_dir = output_path / "meta"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # --- 1. Setup Phase ---
    print(f"Opening HDF5 file at {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        episode_keys = sorted([key for key in f.keys() if key.startswith('demo_')])
        if not episode_keys:
            raise ValueError("No episodes found in HDF5 file.")
        
        # Detect types from first episode
        first_grp = f[episode_keys[0]]
        if 'points' in first_grp:
            obs_key, obs_type, task_desc = 'points', 'point_cloud', "act_point_cloud_imitation"
        elif 'rgb' in first_grp:
            obs_key, obs_type, task_desc = 'rgb', 'image', "act_rgb_imitation"
        else:
            raise ValueError("Could not find 'points' or 'rgb' data.")

        # Get shapes for initialization
        sample_obs = first_grp[obs_key][0]
        if obs_type == 'image':
            # Convert (H, W, C) -> (C, H, W)
            sample_obs = sample_obs.transpose(2, 0, 1)
        
        obs_shape = sample_obs.shape
        state_shape = first_grp['path'][0].shape
        action_shape = first_grp['path'][0].shape
        obs_dtype = str(sample_obs.dtype)

        print(f"Detected {obs_type} with shape {obs_shape}")

        # Initialize Streaming Stats trackers
        stats_trackers = {
            f"observation.{obs_type}": StreamingStats(obs_shape, is_image=(obs_type=='image')),
            "observation.state": StreamingStats(state_shape),
            "action": StreamingStats(action_shape),
            "timestamp": StreamingStats((1,))
        }

        # Setup Parquet Writer
        parquet_path = data_dir / "episodes.parquet"
        # Define Schema matches the dict structure we create below
        # Note: PyArrow infers types well, but explicit schema is safer for images. 
        # We will let pandas infer schema from the first chunk to keep code simple.
        writer = None 
        
        buffer = []
        BUFFER_SIZE = 500 # Adjust based on your RAM (500 frames is usually safe)
        
        all_episodes_metadata = []
        total_frames = 0
        
        # --- 2. Processing Loop ---
        print(f"Processing {len(episode_keys)} episodes...")
        
        for episode_idx, group_name in enumerate(episode_keys):
            episode_group = f[group_name]
            # Load only current episode into memory
            raw_obs = episode_group[obs_key][:]
            raw_path = episode_group['path'][:]
            num_steps = raw_path.shape[0]

            episode_meta = {
                "episode_index": episode_idx,
                "tasks": [task_desc],
                "length": num_steps
            }
            all_episodes_metadata.append(episode_meta)

            for t in range(num_steps):
                # 1. Prepare Data
                step_obs = raw_obs[t]
                if obs_type == 'image':
                    step_obs = step_obs.transpose(2, 0, 1) # HWC -> CHW
                
                step_state = raw_path[t]
                step_action = raw_path[t] # Assuming state=action as per your original script
                
                # 2. Update Statistics (Streaming)
                stats_trackers[f"observation.{obs_type}"].update(step_obs)
                stats_trackers["observation.state"].update(step_state)
                stats_trackers["action"].update(step_action)
                stats_trackers["timestamp"].update(np.array([float(t)]))

                # 3. Create Row
                row = {
                    f"observation.{obs_type}": step_obs.tolist(), # Parquet handles lists
                    "observation.state": step_state.tolist(),
                    "action": step_action.tolist(),
                    "task_index": [0],
                    "episode_index": [episode_idx],
                    "frame_index": t,
                    "timestamp": [float(t)],
                    "next.done": (t == num_steps - 1),
                }
                buffer.append(row)
                total_frames += 1

                # 4. Flush Buffer if full
                if len(buffer) >= BUFFER_SIZE:
                    df_chunk = pd.DataFrame(buffer)
                    table = pa.Table.from_pandas(df_chunk)
                    
                    if writer is None:
                        # Initialize writer with schema from first chunk
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    
                    writer.write_table(table)
                    buffer = [] # Clear memory
            
            if (episode_idx + 1) % 10 == 0:
                print(f"Processed {episode_idx + 1} episodes...")

        # Flush remaining buffer
        if buffer:
            df_chunk = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            writer.write_table(table)
        
        if writer:
            writer.close()

    print(f"Data saved to {parquet_path}")

    # --- 3. Save Metadata (info.json, stats.json, etc) ---
    print("Generating metadata...")

    # Info.json
    info_dict = {
        "codebase_version": "v2.0",
        "total_episodes": len(episode_keys),
        "total_frames": total_frames,
        "chunks_size": 64, # Default chunk size for training
        "fps": 30,
        "splits": {"train": f"0:{len(episode_keys)}"},
        "data_path": "data/episodes.parquet",
        "features": {
            f"observation.{obs_type}": {
                "dtype": "image" if obs_type == 'image' else obs_dtype,
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
    
    if obs_type == 'image':
        info_dict["features"][f"observation.{obs_type}"]["names"] = ["channel", "height", "width"]

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info_dict, f, indent=4)

    # Tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": task_desc}) + '\n')

    # Episodes.jsonl
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for item in all_episodes_metadata:
            f.write(json.dumps(item) + '\n')

    # Stats.json (Retrieve from our StreamingStats objects)
    print("Finalizing statistics...")
    stats_dict = {}
    for key, tracker in stats_trackers.items():
        stats_dict[key] = tracker.get_final_stats()

    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=4)

    print("\n--- LeRobot Dataset creation complete! ---")
    print(f"Your dataset is ready at: {output_path}")

if __name__ == "__main__":
    # Ensure you have pyarrow installed: pip install pyarrow pandas h5py numpy
    my_data_file = "./demos/table_demo.h5"
    local_save_path = "./lerobot_datasets/table_demo"
    
    # Safety check
    if os.path.exists(local_save_path):
        import shutil
        print(f"Warning: Cleaning up existing directory {local_save_path}")
        shutil.rmtree(local_save_path)

    convert_h5_to_final_lerobot_format(my_data_file, local_save_path)