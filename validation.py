import gymnasium as gym
import time

# create car racing v3 environment
env = gym.make('CarRacing-v3', render_mode='human')

def test_car_racing_env():
    """
    create and render the car racing v3 environment with an agent that takes random actions at each step
    (checks if environment is working)
    """
    try:
        env.reset()
        done = False
        while not done:
            # take a random action from the action space
            action = env.action_space.sample()
            # perform the action in the environment
            observation, reward, done, truncated, info = env.step(action)
            
            # Render the environment to visually check if it's working
            env.render()
            
            # Add a short delay to slow down the rendering
            time.sleep(0.01)
        
        print("car racing v3 environment working")

    except Exception as e:
        print(f"error: {e}")

    finally:
        # close environment
        env.close()

if __name__ == "__main__":
    test_car_racing_env()
