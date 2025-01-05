import gymnasium as gym
import pygame
import numpy as np
import time

# create the car racing v3 environment
env = gym.make('CarRacing-v3', render_mode='human')

def process_input():
    """
    Capture user input and map to CarRacing-v3 action space.
    Returns a list [steering, acceleration, brake].
    """
    keys = pygame.key.get_pressed()
    
    # default action: do nothing
    action = [0.0, 0.0, 0.0]
    
    if keys[pygame.K_UP]:      # accelerate
        action[1] = 0.6
    # if keys[pygame.K_DOWN]:    # brake
    #     action[2] = 1.0
    if keys[pygame.K_LEFT]:    # left
        action[0] = -0.3
        action[2] = 0.1
    if keys[pygame.K_RIGHT]:   # right
        action[0] = 0.3
        action[2] = 0.1
    
    return np.array(action, dtype=np.float32)  # convert to np array

def play_car_racing():
    try:
        env.reset()
        pygame.init()
        clock = pygame.time.Clock()

        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # process keyboard input for actions
            action = process_input()
            
            # perform action in the environment
            observation, reward, done, truncated, info = env.step(action)
            
            # render environment
            env.render()
            
            # limit to 60 fps
            clock.tick(60)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # close environment and pygame
        env.close()
        pygame.quit()

if __name__ == "__main__":
    play_car_racing()


env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
env
