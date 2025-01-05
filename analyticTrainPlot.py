import re
import matplotlib.pyplot as plt

# read the log data from the file
with open("results.txt", "r") as file:
    log_data = file.read()

# parse the log data for episode and reward values
pattern = r"Episode (\d+)/\d+ \| Reward: ([\d\.\-]+)"
matches = re.findall(pattern, log_data)

# extract episodes and rewards from each line
episodes = [int(match[0]) for match in matches]
rewards = [float(match[1]) for match in matches]

# calculate average rewards every 50 episodes (same as model save frequency)
window_size = 50
average_rewards = [
    sum(rewards[i:i + window_size]) / len(rewards[i:i + window_size]) 
    for i in range(0, len(rewards), window_size)
]

average_episodes = [
    sum(episodes[i:i + window_size]) // len(episodes[i:i + window_size]) 
    for i in range(0, len(episodes), window_size)
]

# highest reward and its episode
highest_reward = max(rewards)
highest_episode = episodes[rewards.index(highest_reward)]

# highest average reward and its episode
highest_avg_reward = max(average_rewards)
highest_avg_episode = average_episodes[average_rewards.index(highest_avg_reward)]

# plot the rewards
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, label="Rewards", alpha=0.7)
plt.plot(average_episodes, average_rewards, label="Average Rewards (50 episodes)", linestyle='--', linewidth=2)

# add a horizontal line for the highest score
plt.axhline(y=highest_reward, color='red', linestyle='-.', label=f"Highest Reward: {highest_reward:.2f} (Episode {highest_episode})")
# highlight highest average score episode
plt.scatter([highest_avg_episode], [highest_avg_reward], color='red', zorder=5, label=f"Highest Avg. Reward Episode {highest_avg_episode}")

# labels and title
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward vs Episodes with 50-Episode Averages")
plt.legend()
plt.grid()
plt.show()
