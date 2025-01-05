import gymnasium as gym
import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.losses import MeanSquaredError
import argparse

custom_loss = {
    'mse': MeanSquaredError()
}

# mapped discrete actions (same as stacked training)
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),   # do nothing
    np.array([-0.3, 0.0, 0.1]),  # steer left with minor brake
    np.array([0.3, 0.0, 0.1]),   # steer right with minor brake
    np.array([0.0, 0.5, 0.0]),   # accelerate
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)

# frame stack size (same as training: 4)
STACK_SIZE = 4



def preprocess_frame(frame):
    """
    preprocess an rgb frame: convert to grayscale, center crop to 64x64, normalise to [0,1]
    @param frame: an RGB image (np array) of shape (96, 96, 3)
    @return: a preprocessed grayscale image of shape (64, 64)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[16:80, 16:80]  # centre crop to 64x64
    normalized = cropped / 255.0

    return normalized


def create_initial_stack(frame, stack_size=4):
    """
    create an initial stack of frames by repeating the first frame.
    @param frame: a preprocessed frame of shape (64, 64)
    @param stack_size: number of frames to stack
    @return: stacked frames of shape (64, 64, stack_size)
    """

    return np.stack([frame] * stack_size, axis=-1)


def update_stack(stack, new_frame):
    """
    update the frame stack by removing the oldest frame and adding the new frame
    @param stack: current stacked frames of shape (64, 64, 4)
    @param new_frame: new preprocessed frame of shape (64, 64)
    @return: updated stacked frames of shape (64, 64, 4)
    """
    return np.concatenate([stack[:, :, 1:], np.expand_dims(new_frame, axis=-1)], axis=-1)


def evaluate_ddqn(
    model_path='ddqn_speed_1450.h5',
    env_name='CarRacing-v3',
    continuous=True, 
    n_eval_episodes=50, 
    max_steps=1000, 
    render=True, 
    frame_skip=4 
):
    """
    evaluate a trained Frame Stacking DDQN model on the specified gym environment
    @param model_path: path to the saved model
    @param env_name: gym environment
    @param continuous: if continuous action space
    @param n_eval_episodes: number of episodes to evaluate for
    @param max_steps: max steps per episode (same as environment max: 1000)
    @param render: if render game
    @param frame_skip: number of frames to skip (where action repeats for frames skipped)
    """
    # create environment
    env = gym.make(env_name, continuous=continuous, render_mode="human")

    # load trained model
    model = load_trained_model(model_path)

    # store total rewards for each episode
    episode_rewards = []

    # iterate through evaluation episdoes
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_stack = create_initial_stack(frame, stack_size=STACK_SIZE)

        episode_reward = 0.0

        # iterate through steps of each episode
        for step in range(max_steps):
            if render:
                env.render()

            # prepare state for prediction: (1, 64, 64, 4)
            state_input = np.expand_dims(state_stack, axis=0)

            # predict Q values from the model
            q_values = model.predict(state_input, verbose=0)  # (1, NUM_ACTIONS)
            # choose action with highest Q value
            action_idx = np.argmax(q_values[0])  
            action = DISCRETE_ACTIONS[action_idx]

            # execute best action with frame skipping
            accumulated_reward = 0.0
            done = False
            truncated = False

            # check current step and action
            print("Step:", step, ", Action:", action_idx)

            for _ in range(frame_skip):
                next_obs, reward, done, truncated, info = env.step(action)
                accumulated_reward += reward

                if done or truncated:
                    break

            # preprocess next frame
            if not (done or truncated):
                next_frame = preprocess_frame(next_obs)
                state_stack = update_stack(state_stack, next_frame)
            else:
                # if episode is done, no need to update the stack
                pass

            episode_reward += accumulated_reward

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"[Eval] Episode {episode+1}/{n_eval_episodes} - Reward: {episode_reward:.2f}")
        

    env.close()


    # calculate and print average reward
    avg_reward = np.mean(episode_rewards)
    max_reward = max(episode_rewards)
    print(f"\n=== Evaluation Complete ===")
    print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward:.2f}")
    print(f"Highest Reward over {n_eval_episodes} episodes: {max_reward:.2f}")
    # track coverage


def load_trained_model(model_path):
    """
    load the trained DDQN model
    @param model_path: path to the saved model
    @return: loaded model
    """
    try:
        model = load_model(model_path, custom_objects=custom_loss)
        print(f"Model loaded successfully from {model_path}")

        return model
    
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a trained Frames Stacking DDQN model on CarRacing-v3")
    parser.add_argument('--model_path', type=str, default='dqn_framestack_900.h5', help='path to the trained model (.h5 file)')
    parser.add_argument('--env_name', type=str, default='CarRacing-v3', help='gymnasium environment name')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='render the environment during evaluation')
    parser.add_argument('--frame_skip', type=int, default=4, help='number of frames to skip (action repeats)')
    
    args = parser.parse_args()

    evaluate_ddqn(
        model_path=args.model_path,
        env_name=args.env_name,
        continuous=True,  
        n_eval_episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        render=args.render,
        frame_skip=args.frame_skip
    )

    
