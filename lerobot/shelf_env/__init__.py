# In my_custom_env/__init__.py
from gymnasium.envs.registration import register

register(
     id="ShelfEnv-v0",
     entry_point="lerobot.shelf_env.env:ShelfEnv",
     max_episode_steps=300, # As in the lerobot script
)