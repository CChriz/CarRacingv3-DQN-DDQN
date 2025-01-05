--------------------
Installation & Setup
--------------------

we took the following steps in setting up the environment and producing our results:

Install anaconda: https://www.anaconda.com/download

Create anaconda environment
- navigate to Anaconda Prompt (NOT command prompt) and execute the following scripts:
	conda create -n gymenv
	conda activate gymenv
	conda install python=3.11
	conda install swig
	pip install gymnasium[box2d]
(you may need to install C++ build tools from the link in the error message if not installed before. Ensure you select Desktop Developer with C++ then try again):   
	pip install gymnasium[box2d]


we used Visual Studio Code for developing our code, to run the code:
1. Open our code folder in Visual Studio Code
2. go to [Select Interpreter] and choose "gymenv" - the one created with conda (Ctrl+Shift+P)
3. go to [Terminal Select Default Profile] and choose "Command Prompt" instead of "PowerShell".
4. go to [Files] and open Terminal (command prompt) within VS Code
5. Execute:
	conda activate gymenv


You should now be able to execute validation/training/evaluation scripts:

- to validate the environment can be created and rendered successfully, run the "validation.py" file

- to play the rendered environment yourself (human control), run the "youPlay.py" file	


we recommend following the following YouTube video for installation and setup of the Box2D CarRacing-v3 environment:
https://www.youtube.com/watch?time_continue=429&v=gMgj4pSHLww&embeds_referring_euri=https%3A%2F%2Fdiscord.com%2F&source_ve_path=MjM4NTE


---------------
Assisting Files
---------------
- validation.py : to validate the environment can be created and rendered successfully
- youPlay.py : to play the rendered environment yourself (human control)


-------------------------
Plot Reproducibility File
-------------------------
Training Plots (Rewards vs. Episodes) are automatically generated upon completion of Training Script.
However, to produce the plots as seen in our report (highlighting highest reward episode, average reward line):
- copy entire command line output from the training script, and paste into the "results.txt" file:
e.g.
...
Episode 1949/2000 | Reward: 365.12 | Epsilon: 0.05
Episode 1950/2000 | Reward: 493.33 | Epsilon: 0.05
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.     
Checkpoint saved to ddqn_speed_1950.h5
Episode 1951/2000 | Reward: 596.97 | Epsilon: 0.05
Episode 1952/2000 | Reward: 520.92 | Epsilon: 0.05
Episode 1953/2000 | Reward: 505.54 | Epsilon: 0.05
...

Now execute the "analyticTrainPlot.py" file, which will parse each line in the text file to extract episode and reward values and plot the Reward vs. Episodes graph, and highlighting maximum reward obtained with red line, and orange line for average rewards over every 50 episodes, and maximum average reward obtained with a red dot.

Episode of occurrence for max reward and max average reward denoted in Legend on the side of the plot.


-------------------------------------------
Files - Model Training, Loading, Evaluation
-------------------------------------------

(1) Frame-Stack DDQN agent:
- 1_DDQN_framestack_train.py : script to train a DDQN agent using frame stacking
- 1_DDQN_framestack_load.py : script to load and visualise performance of a pre-trained DDQN agent with frame stacking
- 1_DDQN_framestack_evaluation.py : script to evaluate performance of 14 Frame Stacking DDQN agent checkpoints during training

Saved Frame-Stack DDQN models:
- naming convention: "ddqn_speed_<episode>.h5" : pre-trained model weights at <episode>
- best performing model: "ddqn_checkpoint_1450.h5"


(2) Frame-Stack DQN agent:
- 2_DQN_framestack_train.py : script to train a DDQN agent using frame stacking
- 2_DQN_framestack_load.py : script to load and visualise performance of a pre-trained DQN agent with frame stacking
- 2_DQN_framestack_evaluation.py : script to evaluate performance of 14 Frame Stacking DQN agent checkpoints during training

Saved Frame-Stack DDQN models:
- naming convention: "dqn_framestack_<episode>.h5" : pre-trained model weights at <episode>


(3) Single Frame DDQN agent:
- 3_DDQN_singleframe_train.py : script to train a DDQN agent using single frames
- 3_DDQN_singleframe_load.py : script to load and visualise performance of the pre-trained Single Frame DDQN agent

Saved Single-Frame DDQN models:
- naming convention: "ddqn_model400.h5" : pre-trained model weights at episode 400
(this method was discontinued in favour of stacked-frame models due to severely pro-longed training times and failure to capture temporal features)


---------------
Usage
---------------

---------
TRAINING:
---------

(1) To train the stacked frames DDQN agent, run the DDQN_framestack_load.py file
- by default this trains the agent with the following attributes:
        env_name='CarRacing-v3',
        continuous=True, 	# continuous space but actions mapped to discrete in DISCRETE_ACTIONS
        stack_size=4, 		# number of stacked consecutive frames (used 4)
        frame_skip=4, 		# number of frames skipped per decision (used 4, same as stacked consecutive frames)
        n_episodes=2000, # number of episodes to train agent for
        max_steps=250, 		# max step per episode - only 250 since each decision repeats action for 4 frames (~1000 frames = max environment step)
        batch_size=64, 		# batch size for replay buffer
        gamma=0.99,
        epsilon_start=1.0, 	# initial epsilon value (for epsilon-greedy action choice)
        epsilon_min=0.05, 	# min epsilon value 
        epsilon_decay=0.997, 	# epsilon decay rate
        update_target_freq=500, # TARGET network updates per _ steps (default: 500)
        render=False, 		# if render environment (false by default for training)
        save_freq=20, 		# save model checkpoints every 50 episodes
        checkpoint_prefix='ddqn_speed' # model save path prefix - suffix: episode, e.g. _420
upon training completion, a graph of episode vs total reward is plotted.


(2) To train the stacked frames DQN agent, run the DQN_framestack_load.py file
- by default this trains the agent with the following attributes:
        env_name='CarRacing-v3',
        continuous=True, 	# continuous space but actions mapped to discrete in DISCRETE_ACTIONS
        stack_size=4, 		# number of stacked consecutive frames (used 4)
        frame_skip=4, 		# number of frames skipped per decision (used 4, same as stacked consecutive frames)
        n_episodes=2000, 	# number of episodes to train agent for
        max_steps=250, 		# max step per episode - only 250 since each decision repeats action for 4 frames (~1000 frames = max environment step)
        batch_size=64, 		# batch size for replay buffer
        gamma=0.99,
        epsilon_start=1.0, 	# initial epsilon value (for epsilon-greedy action choice)
        epsilon_min=0.05, 	# min epsilon value 
        epsilon_decay=0.997, 	# epsilon decay rate
        update_target_freq=500, # TARGET network updates per _ steps (default: 500)
        render=False, # if render environment (false by default for training)
        save_freq=20, # save model checkpoints every 50 episodes
        checkpoint_prefix='dqn_framestack' # model save path prefix - suffix: episode, e.g. _420
upon training completion, a graph of episode vs total reward is plotted.


(3) To train the single frame DDQN agent, run the DDQN_train.py file
- by default this trains the agent with the following attributes:
        env_name='CarRacing-v3',
        continuous=True, 	# set to continuous but mapped to discrete with DISCRETE_ACTIONS
        n_episodes=500, 	# total training episodes
        max_steps=1000, 	# max steps per episode
        batch_size=64, 		# replay batch size
        gamma=0.99, 
        epsilon_start=1.0, 	# initial epsilon value (for epsilon-greedy action choice)
        epsilon_min=0.05, 	# min epsilon value 
        epsilon_decay=0.995, 	# epsilon decay rate
        update_target_freq=1000, # rate of TARGET net update
        render=False, # environment render (false by default for training)
        save_freq=10, # save model every 10 episodes
        save_path='ddqn_model.h5' # file name to save model
upon training completion, a graph of episode vs total reward is plotted.

----------------
LOAD & VISUALISE
----------------

(1) To load and visualise a pre-trained stacked frames DDQN agent, run the DDQN_framestack_load.py file
- by default this loads the _ model and renders its performance for 10 episodes. Actions chosen at each step and rewards achieved are printed.

(2) To load and visualise a pre-trained stacked frames DQN agent, run the DQN_framestack_load.py file
- by default this loads the _ model and renders its performance for 10 episodes. Actions chosen at each step and rewards achieved are printed.

(3) To load and visualise a pre-trained single frame DDQN agent, run the DDQN_load.py file
- by default this loads the _ model and renders its performance for 10 episodes. Actions chosen at each step and rewards achieved are printed.

----------
EVALUATION
----------

(1) To Evaluate the 14 Frame-Stack DDQN checkpoints across 2000 episodes
displays average reward and max reward over 50 evaluation episodes of each model

(2) To Evaluate the 14 Frame-Stack DQN checkpoints across 2000 episodes
displays average reward and max reward over 50 evaluation episodes of each model

(3) To Evaluate the best performing single frame DDQN agent at episode 400
displays average reward and max reward over 50 evaluation episodes


explanations & instructions for Modifications 
(i.e. Training Hyperparameters: max episodes, max steps per episode, epsilon decay, etc.)
(i.e. Evaluation: model to load, number of episodes, etc.)
all indicated in detail in comments and docstrings in their respective code files.


Thank you for reading :)
