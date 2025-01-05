import gymnasium as gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from collections import deque

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras._tf_keras.keras.optimizers import Adam

def preprocess_frame(frame):
    """
    Preprocess an RGB frame: grayscale, crop borders, normalise to [0,1]
    @param frame: a 96x96 iamge (np array)
    @return: a 64x64 image (np array)
    """
    # convert to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop to keep the center part (from the original 96x96)
    # rids of bottom border with no map information
    cropped = gray[16:80, 16:80]

    # normalising pixel values to [0, 1]
    normalised = cropped / 255.0

    return normalised


# map continuous action space to 5 discrete actions
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),  # do nothing
    np.array([-0.3, 0.0, 0.0]), # steer left
    np.array([0.3, 0.0, 0.0]),  # steer right
    np.array([0.0, 0.3, 0.0]),  # accelerate
    np.array([0.0, 0.0, 0.3]),  # brake - proved erroneous a lot of the times leading to agent staying still
]

# default: 5 actions
NUM_ACTIONS = len(DISCRETE_ACTIONS)


# replay buffer
class ReplayBuffer:
    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)


    def store(self, state, action, reward, next_state, done):
        """
        store a transition (s, a, r, s', done) in the replay buffer
        @param state: current state, shape: (64, 64, 1)
        @param action: index of discrete action (int)
        @param reward: float
        @param next_state: next state, shape: (64, 64, 1)
        @param done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))


    def sample_batch(self, batch_size=64):
        """
        randomly sample a batch of transitions
        @param: batch_size: number of samples per batch (64 by default)
        @returns: (states, actions, rewards, next_states, dones)
            states: (batch_size, 64, 64, 1)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, 64, 64, 1)
            dones: (batch_size,)
        """
        # randomly sample a batch
        batch = random.sample(self.buffer, batch_size)
        # retrieve states, actions, rewards, next states, dones for all samples in batch
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
    

def build_q_network(input_shape=(64, 64, 1), num_actions=5, learning_rate=1e-3):
    """
    a CNN-based Q-network that outputs Q-values for each discrete action
    @param input_shape: dimensions of input image (state)
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
        # one Q-value per discrete action (default 5)
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


def train_ddqn(online_network, target_network, replay_buffer, batch_size=64, gamma=0.99):
    """
    sample a batch from the replay_buffer and do a single DDQN update step

    DDQN target:
        y = r + gamma * Q_target(s', argmax_a (Q_online(s')))
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)

    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)

    # (1) get best actions to take : actions with max q values over next_states from ONLINE network
    # shape: (batch_size, NUM_ACTIONS)
    next_q_online = online_network(next_states_tf)  
    # shape: (batch_size)
    best_actions = tf.argmax(next_q_online, axis=1)  

    # (2) evaluate the actions with TARGET network
    # shape: (batch_size, NUM_ACTIONS)
    next_q_target = target_network(next_states_tf)  
    # shape: (batch_size)
    best_q_values = tf.reduce_sum(
        next_q_target * tf.one_hot(best_actions, NUM_ACTIONS),
        axis=1
    )

    # (3) build DDQN targets
    #    if done: y = reward
    #    else: y = reward + gamma * Q_target(...)
    targets = rewards + (1 - dones) * gamma * best_q_values

    # (4) FORWARD PASS on ONLINE network for current states
    with tf.GradientTape() as tape:
        # shape: [batch_size, NUM_ACTIONS]
        q_values = online_network(states_tf)  

        # get Q-values for taken actions
        one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
        pred_q = tf.reduce_sum(q_values * one_hot_actions, axis=1)

        # MSE LOSS
        loss = tf.reduce_mean(tf.square(targets - pred_q))

    # (5) BACKPROP
    grads = tape.gradient(loss, online_network.trainable_variables)
    online_network.optimizer.apply_gradients(zip(grads, online_network.trainable_variables))

    return loss.numpy()


def ddqn_train(
    env_name='CarRacing-v3',
    continuous=True,
    n_episodes=500,
    max_steps=1000,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    update_target_freq=1000,
    render=False,
    save_freq=10,
    save_path='ddqn_model.h5'
):
    """
    train a DDQN on CarRacing-v3 environment with discrete actions
    """
    # create environment
    env = gym.make(env_name, continuous=continuous)
    obs, _ = env.reset()

    # build ONLINE and TARGET networks
    online_q = build_q_network(input_shape=(64, 64, 1), num_actions=NUM_ACTIONS)
    target_q = build_q_network(input_shape=(64, 64, 1), num_actions=NUM_ACTIONS)
    
    # clone TARGET networks to be the same as ONLINE network initially
    # (so same initial weights)
    target_q.set_weights(online_q.get_weights())  

    # create replay buffer
    replay_buffer = ReplayBuffer(max_size=50_000)
    # training stats - for plots and evaluations
    epsilon = epsilon_start
    episode_rewards = []
    total_steps = 0

    # training episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        # shape (64,64)
        state = preprocess_frame(obs)
        # shape (64,64,1)
        state = np.expand_dims(state, axis=-1)

        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1

            # eps-greedy action
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(NUM_ACTIONS)
            else:
                q_vals = online_q.predict(np.array([state]), verbose=0)  # shape [1, NUM_ACTIONS]
                action_idx = np.argmax(q_vals[0])

            # execute action in environment
            action = DISCRETE_ACTIONS[action_idx]
            next_obs, reward, done, truncated, info = env.step(action)

            if render:
                env.render()

            # preprocess next observation
            next_state = preprocess_frame(next_obs)
            next_state = np.expand_dims(next_state, axis=-1)

            episode_reward += reward

            # store transition
            replay_buffer.store(state, action_idx, reward, next_state, done or truncated)

            # move to next state
            state = next_state

            # TRAIN STEP (if enough samples)
            if replay_buffer.size() >= batch_size:
                loss = train_ddqn(online_q, target_q, replay_buffer, batch_size=batch_size, gamma=gamma)

            # UPDATE TARGET network
            if total_steps % update_target_freq == 0:
                target_q.set_weights(online_q.get_weights())

            if done or truncated:
                break

        # decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        # print current episode and reward obtained
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{n_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

        # SAVE MODEL every save_freq episodes
        if (episode + 1) % save_freq == 0:
            online_q.save(save_path)
            print(f"Model saved to {save_path} at episode {episode+1}")

    env.close()
    return episode_rewards



# MAIN CODE - run training
if __name__ == "__main__":
    rewards = ddqn_train(
        env_name='CarRacing-v3',
        continuous=True, # set to continuous but mapped to discrete with DISCRETE_ACTIONS
        n_episodes=500, # total training episodes
        max_steps=1000, # max steps per episode
        batch_size=64, # replay batch size
        gamma=0.99, 
        epsilon_start=1.0, # initial eps
        epsilon_min=0.05, # min eps (random actions)
        epsilon_decay=0.995, # eps decay rate
        update_target_freq=1000, # rate of TARGET net update
        render=False, # environment render (false by default for training)
        save_freq=10, # save model every 10 episodes
        save_path='ddqn_model.h5' # file name to save model
    )

    # plot the rewards over episodes
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDQN CarRacing-v3 Training")
    plt.show()