import gymnasium as gym
import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.losses import MeanSquaredError

custom_loss = {
    'mse': MeanSquaredError()
}


def preprocess_frame(frame):
    """
    preprocess an RGB frame: grayscale, crop borders, normalise to [0,1]
    @param frame: a 96x96 image (np array)
    @return: a 64x64 image (np array)
    """
    # convert to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop to keep the center part (from the original 96x96)
    # remove top/bottom/side borders
    cropped = gray[16:80, 16:80]

    # normalising pixel values to [0, 1]
    normalised = cropped / 255.0

    return normalised

# brake
# DISCRETE_ACTIONS = [
#     np.array([0.0, 0.0, 0.0]),  # do nothing
#     np.array([-0.3, 0.0, 0.0]), # steer left
#     np.array([0.3, 0.0, 0.0]),  # steer right
#     np.array([0.0, 0.3, 0.0]),  # accelerate
#     np.array([0.0, 0.0, 0.3]),  # brake
# ]

# no brake - less acceleration - solves staying still
# DISCRETE_ACTIONS = [
#     np.array([0.0, 0.0, 0.0]),  # do nothing
#     np.array([-0.3, 0.0, 0.0]), # steer left
#     np.array([0.3, 0.0, 0.0]),  # steer right
#     np.array([0.0, 0.3, 0.0]),  # accelerate
#     np.array([0.0, 0.2, 0.0]),  # less accelerate
# ]

# no pure brake - brake while steering - avoid skidding
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),  # do nothing
    np.array([-0.3, 0.0, 0.1]), # steer left
    np.array([0.3, 0.0, 0.1]),  # steer right
    np.array([0.0, 0.4, 0.0]),  # accelerate
    np.array([0.0, 0.2, 0.0]),  # less accelerate
]



NUM_ACTIONS = len(DISCRETE_ACTIONS)

def load_trained_model(model_path):
    """
    loads the trained DDQN model from disk.
    @param model_path: path to the .h5 file
    @return: trained tf.keras model
    """
    model = load_model(model_path, custom_objects=custom_loss)
    return model


def evaluate_ddqn(model_path='ddqn_model400.h5', 
                  env_name='CarRacing-v3', 
                  continuous=True, 
                  n_eval_episodes=10, 
                  max_steps=1000, 
                  render=False):
    """
    evaluate a saved DDQN model on the CarRacing-v3 environment.

    @param model_path: path to the saved .h5 model (default 'ddqn_model.h5')
    @param env_name: name of the Gymnasium environment (default 'CarRacing-v3')
    @param continuous: whether the environment is continuous (default True)
    @param n_eval_episodes: number of evaluation episodes
    @param max_steps: max steps per episode
    @param render: whether to render the environment

    prints average reward after all episodes
    """
    # load the environment
    env = gym.make(env_name, continuous=continuous, render_mode="human")

    # load the trained model
    model = load_trained_model(model_path)

    # store total rewards for each episode
    episode_rewards = []

    # run evaluation episodes
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        state = preprocess_frame(obs)
        state = np.expand_dims(state, axis=-1)  # shape (64,64,1)
        
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()

            epsilon = 0.13
            # eps-greedy action
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(NUM_ACTIONS)

            else:
                # get Q-values from the loaded model
                q_vals = model.predict(np.array([state]), verbose=0)  # shape [1, NUM_ACTIONS]
                action_idx = np.argmax(q_vals[0])
                print(f"Step {step}: Q-values = {np.argmax(q_vals[0])}")
            
            # convert discrete action index to actual environment action
            action = DISCRETE_ACTIONS[action_idx]
            
            # next step in the environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # preprocess next state
            next_state = preprocess_frame(next_obs)
            next_state = np.expand_dims(next_state, axis=-1)
            
            episode_reward += reward
            
            # go to next state
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"[Eval] Episode {episode+1}/{n_eval_episodes} - Reward: {episode_reward:.2f}")

    env.close()

    # Print average reward
    avg_reward = np.mean(episode_rewards)
    print(f"\n=== Evaluation Complete ===")
    print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    # example usage:
    evaluate_ddqn(
        model_path='ddqn_model400.h5',
        env_name='CarRacing-v3',
        continuous=True,
        n_eval_episodes=10,
        max_steps=1000,
        render=True
    )
