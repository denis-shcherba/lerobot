import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Import your custom environment package to register it
import lerobot.shelf_env.env  # noqa: F401 

print("Attempting to create the environment: ShelfEnv-v0")
# 1. Create an instance of your environment
try:
    env = gym.make("ShelfEnv-v0")  # Adjust args as needed
    print("✅ Environment created successfully!")
except Exception as e:
    print(f"❌ Failed to create environment: {e}")
    exit()

# 2. Run the official checker
print("\nRunning the official Gymnasium environment checker...")
try:
    check_env(env.unwrapped)
    print("✅ Passed the environment check!")
    print("Your environment is compliant with the Gymnasium API.")
except Exception as e:
    print("❌ The environment checker raised an exception:")
    print(e)

env.close()