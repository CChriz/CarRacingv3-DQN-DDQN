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
    convert image (frame) to to grayscale, centre crop from 96x96 to 64x64, normalise to [0,1]
    @returns: preprocessed frame of shape (64,64)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[16:80, 16:80]  # a 64x64 image
    normalized = cropped / 255.0

    return normalized


# DISCRETE ACTION MAPPING
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),  # do nothing
    np.array([-0.3, 0.0, 0.1]), # steer left
    np.array([0.3, 0.0, 0.1]),  # steer right
    np.array([0.0, 0.5, 0.0]),  # accelerate
    # brake removed due to previous braking issues
    # (minor brake applied automatically simultaneously with turning)
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)


# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        """
        store a transition (state, action, reward, next_state, done).
            state: (64,64,4)
            next_state: (64,64,4)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size=64):
        """
        randomly sample a batch from the replay buffer
        @param: 
        @returns: (states, actions, rewards, next_states, dones)
            states: (batch_size,64,64,4)
            next_states: (batch_size,64,64,4)
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



# CNN-BASED Q NET
def build_q_network(input_shape=(64, 64, 4), num_actions=4, learning_rate=1e-3):
    """
    a CNN-based Q-network that outputs Q-values for each discrete action
    @param input_shape: dimensions of input image (state) size x number of consecutive stacked frames (default 4)
    @param num_actions: number of discrete actions (action)
    @oaram learning_rate: learning rate for adam optimiser
    """
    model = Sequential([
        Input(shape=input_shape),

        # convlution block 1
        # 32 3x3 filters, relu activation
        # 2x2 maxpooling, downsample features
        # 20% dropout regularisation
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # convolution block 2
        # 64 3x3 filters, relu activation
        # 2x2 maxpooling, downsample features
        # 20% dropout regularisation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # fully connected layers
        # flatten 2d features to 1d vector
        # dense downsize to 256, relu activation
        # 20 % dropout regularisation
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),

        # output layer
        # one Q-value per discrete action (default 4)
        # no activation
        Dense(num_actions, activation='linear')
    ])

    model.compile(
        # adam optimiser
        # mse loss
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model



# TRAINING DDQN
def train_ddqn(online_network, target_network, replay_buffer, batch_size=64, gamma=0.99):
    """
    DDQN:
        y = r + gamma * Q_target(s', argmax_a Q_online(s'))
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)

    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)

    # (1) argmax over next_states from ONLINE Q net
    next_q_online = online_network(next_states_tf)  # shape: [batch_size, NUM_ACTIONS]
    best_actions = tf.argmax(next_q_online, axis=1) # [batch_size]

    # (2) evaluate these actions with TAGET Q net
    next_q_target = target_network(next_states_tf)  # shape: [batch_size, NUM_ACTIONS]
    best_q_values = tf.reduce_sum(
        next_q_target * tf.one_hot(best_actions, NUM_ACTIONS),
        axis=1
    )

    # (3) build DDQN TARGETS
    targets = rewards + (1 - dones) * gamma * best_q_values

    # (4) FORWARD PASS on ONLINE Q for current states
    with tf.GradientTape() as tape:
        q_values = online_network(states_tf)  # [batch_size, NUM_ACTIONS]
        one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
        pred_q = tf.reduce_sum(q_values * one_hot_actions, axis=1)
        loss = tf.reduce_mean(tf.square(targets - pred_q))

    # (5) BACKPROP
    grads = tape.gradient(loss, online_network.trainable_variables)
    online_network.optimizer.apply_gradients(zip(grads, online_network.trainable_variables))

    return loss.numpy()



# FRAME STACKING
def create_initial_stack(obs, stack_size=4):
    """
    replicate a single frame obs into a stack of shape (64,64,stack_size).
    """
    stacked = np.stack([obs]*stack_size, axis=-1)  # shape: (64,64,4)
    return stacked


def update_stack(stack, new_frame):
    """
    shift stack left and add new_frame as the last channel.
    stack: (64,64,4), new_frame: (64,64)
    """
    new_stack = np.concatenate([stack[:,:,1:], np.expand_dims(new_frame, axis=-1)], axis=-1)
    return new_stack


# FRAME SKIPPING
def step_with_skip(env, action, skip=4):
    """
    repeat ACTION for SKIPPED frames. sum rewards, return last frame
    (if the episode ends mid-skip, stop and return)
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


# DDQN TRAINING LOOP
def ddqn_train(
    env_name='CarRacing-v3',
    continuous=True,
    stack_size=4,   # frames to stack (default 4)
    frame_skip=4,   # frames to skip (identical to stacked, default 4)
    n_episodes=2000, # episodes to train agent for
    max_steps=250,  # agent decisions per episode (x4 frames = ~1000 frames)
    batch_size=64,  # replay buffer batch size
    gamma=0.99,
    epsilon_start=1.0, 
    epsilon_min=0.05, # minimum eps
    epsilon_decay=0.997, # eps (random exploration) rate of decay
    update_target_freq=500, # rate of target net update per steps
    render=False, # if render (false for training by default)
    save_freq=50, # save model checkpoint per training episdoes (default 20)
    checkpoint_prefix='ddqn_speed' # model save name prefix - suffix: episode - e.g. _420
):
    """
    train a DDQN with:
        frame stacking (stack_size=4)
        frame skipping (frame_skip=4)
        save checkpoints every 20 episodes
        max 250 agent decisions (each repeats for 4 frames, total 1000)
    """

    # create environment
    env = gym.make(env_name, continuous=continuous)
    obs, _ = env.reset()

    # build Q nets
    online_q = build_q_network(input_shape=(64,64,stack_size), num_actions=NUM_ACTIONS)
    target_q = build_q_network(input_shape=(64,64,stack_size), num_actions=NUM_ACTIONS)
    target_q.set_weights(online_q.get_weights())

    # replay buffer
    replay_buffer = ReplayBuffer(max_size=50_000)

    # training stats
    epsilon = epsilon_start
    episode_rewards = []
    total_steps = 0

    # iterate through training episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)  # shape (64,64)
        state_stack = create_initial_stack(frame, stack_size=stack_size)  # (64,64,4)

        episode_reward = 0.0

        # iterate through each step in the episode
        for step in range(max_steps):
            total_steps += 1

            # eps-greedy action decision (random exploration)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(NUM_ACTIONS)

            else:
                q_vals = online_q.predict(state_stack[np.newaxis,:], verbose=0)  # [1, NUM_ACTIONS]
                action_idx = np.argmax(q_vals[0])

            action = DISCRETE_ACTIONS[action_idx]

            # frame skipping
            next_obs, accumulated_reward, done, truncated, info = step_with_skip(env, action, skip=frame_skip)

            if next_obs is None:
                # if the episode ended mid-skip without a final obs
                episode_reward += accumulated_reward
                break

            if render:
                env.render()

            next_frame = preprocess_frame(next_obs)
            next_state_stack = update_stack(state_stack, next_frame)

            episode_reward += accumulated_reward

            # store state transition in replay buffer
            replay_buffer.store(state_stack, action_idx, accumulated_reward, next_state_stack, done or truncated)

            state_stack = next_state_stack

            # train model if enough samples
            if replay_buffer.size() >= batch_size:
                loss = train_ddqn(online_q, target_q, replay_buffer, batch_size=batch_size, gamma=gamma)

            # update TARGET net
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

        # SAVE checkpoint every save_freq (default 20) episodes
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f"{checkpoint_prefix}_{episode+1}.h5"
            online_q.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    env.close()

    return episode_rewards


# MAIN
if __name__ == "__main__":
    rewards = ddqn_train(
        env_name='CarRacing-v3',
        continuous=True, # continuous space but actions mapped to discrete in DISCRETE_ACTIONS
        stack_size=4, # number of stacked consecutive frames (used 4)
        frame_skip=4, # number of frames skipped per decision (used 4, same as stacked consecutive frames)
        n_episodes=2000, # number of episodes to train agent for
        max_steps=250, # max step per episdoe - only 250 since each decision repeats action for 4 frames (~1000 frames = max environment step)
        batch_size=64, # batch size for replay buffer
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997, # rate of which eps decays (0.995: min at ~600 episodes, 0.997: min at ~1000 episdoes)
        update_target_freq=500, # TARGET network updates per _ steps (default: 500)
        render=False, # if render environment (false by default for training)
        save_freq=50, # save model checkpoints every _ episodes
        checkpoint_prefix='ddqn_speed' # model save path prefix - suffix: episode, e.g. _420
    )
    
    # plot training results: episode vs reward
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDQN + frame stack (CarRacing-v3)")
    plt.show()
