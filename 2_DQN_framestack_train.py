import gymnasium as gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from collections import deque

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten
)
from keras._tf_keras.keras.callbacks import TensorBoard
from keras._tf_keras.keras.optimizers import Adam


# PREPROCESS
def preprocess_frame(frame):
    """
    convert image (frame) to grayscale, center-crop from 96x96 to 64x64, and normalise to [0,1]
    @return np.array: preprocessed frame of shape (64,64)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[16:80, 16:80]  # a 64x64 image
    normalized = cropped / 255.0
    return normalized


DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),  # do nothing
    np.array([-0.3, 0.0, 0.1]), # steer left + mild brake
    np.array([0.3, 0.0, 0.1]),  # steer right + mild brake
    np.array([0.0, 0.5, 0.0]),  # accelerate
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)


# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        """
        store a transition (state, action, reward, next_state, done).
        @params:
            state (np.array): shape (64,64,4)
            action (int): discrete action index
            reward (float)
            next_state (np.array): shape (64,64,4)
            done (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size=64):
        """
        randomly sample a batch from the replay buffer.

        @param batch_size (int)
        @returns:
            tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def size(self):
        return len(self.buffer)


# CNN-BASED Q NETWORK
def build_q_network(input_shape=(64, 64, 4), num_actions=4, learning_rate=1e-3):
    """
    A CNN-based Q-network that outputs Q-values for each discrete action
    @params:
        input_shape (tuple): dimensions of input image (stack_size frames)
        num_actions (int): number of discrete actions
        learning_rate (float): learning rate for Adam optimiser
    @return keras.Model: Q net
    """
    model = Sequential([
        Input(shape=input_shape),

        # convolution block 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # convolution block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),

        # output layer
        Dense(num_actions, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model


# DQN TRAINING
def train_dqn(online_network, target_network, replay_buffer, batch_size=64, gamma=0.99):
    """
    standard DQN update step.
        y = r + gamma * max_a Q_target(s', a)

    @params:
        online_network (keras.Model): network for training
        target_network (keras.Model): network for target Q-value computation
        replay_buffer (ReplayBuffer): replay buffer to sample from
        batch_size (int): batch size for sampling from replay buffer
        gamma (float): discount factor

    @return:
        loss (float)
    """
    # sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)

    # convert to tensors
    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)

    # (1) get Q values for the next states from the TARGET network
    next_q_target = target_network(next_states_tf)  # shape: [batch_size, NUM_ACTIONS]
    # (2) take the max over actions
    best_q_values = tf.reduce_max(next_q_target, axis=1)  # [batch_size]

    # (3) build DQN targets
    targets = rewards + (1 - dones) * gamma * best_q_values

    # (4) forward pass on ONLINE Q for the current states
    with tf.GradientTape() as tape:
        q_values = online_network(states_tf)  # [batch_size, NUM_ACTIONS]
        one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
        pred_q = tf.reduce_sum(q_values * one_hot_actions, axis=1)
        loss = tf.reduce_mean(tf.square(targets - pred_q))

    # (5) backprop
    grads = tape.gradient(loss, online_network.trainable_variables)
    online_network.optimizer.apply_gradients(zip(grads, online_network.trainable_variables))

    return loss.numpy()


# FRAME STACKING
def create_initial_stack(obs, stack_size=4):
    """
    replicates a single frame obs into a stack of shape (64,64,stack_size)
    """
    stacked = np.stack([obs]*stack_size, axis=-1)  # shape: (64,64,4)
    return stacked


def update_stack(stack, new_frame):
    """
    shift stack left and add new_frame as the last channel.
    stack: (64,64,4), new_frame: (64,64)
    """
    new_stack = np.concatenate([stack[:, :, 1:], np.expand_dims(new_frame, axis=-1)], axis=-1)
    return new_stack


# FRAME SKIPPING
def step_with_skip(env, action, skip=4):
    """
    repeat 'action' for skipped frames. Sum rewards and return last frame
    if the episode ends mid-skip: stop and return immediately
    """
    total_reward = 0.0
    done = False
    truncated = False
    last_obs = None

    for _ in range(skip):
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        last_obs = obs
        if done or truncated:
            break

    return last_obs, total_reward, done, truncated, info


# DQN TRAINING
def dqn_train(
    env_name='CarRacing-v3',
    continuous=True,
    stack_size=4,      # frames to stack
    frame_skip=4,      # frames to skip
    n_episodes=600,    # number of episodes
    max_steps=250,     # max steps (agent decisions) per episode
    batch_size=64,     
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    update_target_freq=1000,
    render=False,
    save_freq=20,
    checkpoint_prefix='dqn_checkpoint'
):
    """
    train a DQN with:
        frame stacking (stack_size)
        frame skipping (frame_skip)
        saving checkpoints every 'save_freq' episodes
        each episode has up to 'max_steps' agent decisions
    """
    # create environment
    env = gym.make(env_name, continuous=continuous)
    obs, _ = env.reset()

    # build Q networks
    online_q = build_q_network(input_shape=(64, 64, stack_size), num_actions=NUM_ACTIONS)
    target_q = build_q_network(input_shape=(64, 64, stack_size), num_actions=NUM_ACTIONS)
    target_q.set_weights(online_q.get_weights())

    # replay buffer
    replay_buffer = ReplayBuffer(max_size=50_000)

    # training stats
    epsilon = epsilon_start
    episode_rewards = []
    total_steps = 0

    # training loop
    for episode in range(n_episodes):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_stack = create_initial_stack(frame, stack_size=stack_size)

        episode_reward = 0.0

        for step in range(max_steps):
            total_steps += 1

            # eps-greedy action selection
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(NUM_ACTIONS)

            else:
                q_vals = online_q.predict(state_stack[np.newaxis, :], verbose=0)
                action_idx = np.argmax(q_vals[0])

            action = DISCRETE_ACTIONS[action_idx]

            # frame skipping
            next_obs, accumulated_reward, done, truncated, info = step_with_skip(env, action, skip=frame_skip)

            if next_obs is None:
                # episode ended mid-skip without final obs
                episode_reward += accumulated_reward
                break

            if render:
                env.render()

            next_frame = preprocess_frame(next_obs)
            next_state_stack = update_stack(state_stack, next_frame)

            episode_reward += accumulated_reward

            # store transition in replay buffer
            replay_buffer.store(state_stack, action_idx, accumulated_reward, next_state_stack, done or truncated)
            state_stack = next_state_stack

            # train model if enough samples in replay buffer
            if replay_buffer.size() >= batch_size:
                loss = train_dqn(online_q, target_q, replay_buffer, batch_size=batch_size, gamma=gamma)

            # UPDATE TARGET net
            if total_steps % update_target_freq == 0:
                target_q.set_weights(online_q.get_weights())

            if done or truncated:
                break

        # eps decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{n_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

        # save checkpoint every 'save_freq' episodes
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f"{checkpoint_prefix}_{episode+1}.h5"
            online_q.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    env.close()
    return episode_rewards


# MAIN
if __name__ == "__main__":
    rewards = dqn_train(
        env_name='CarRacing-v3',
        continuous=True,  
        stack_size=4,     
        frame_skip=4,     
        n_episodes=2000,  
        max_steps=250,    
        batch_size=64,    
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        update_target_freq=500,
        render=False,
        save_freq=50,
        checkpoint_prefix='dqn_framestack'
    )
    
    # plot training results: episode vs. reward
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN + Frame Stack (CarRacing-v3)")
    plt.show()
