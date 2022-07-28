import numpy as np
import os
from NeuralNetworks import NeuralNetwork, NNLayer
from ReinforcementLearningPlot import Plotter

class ReinforcementLearning:
    def __init__(self, name, environment, replay_buffer_size):
        self.name = name

        # Environment
        self.environment = environment

        # Estimators
        self.Det_policy = None  # Deterministic policy estimator (used during testing)
        self.Sto_policy = None  # Stochastic policy estimator (used during training)
        self.Q_function = None  # Action-value function estimator
        self.V_function = None  # Value function estimator
        self.V_target = None    # Value function target network
        self.__alpha = None     # Alpha network

        # Algorithm dependant internal variables
        self.__Q_loss, self.__V_loss, self.__folder = None, None, ""

        # Training hyper-parameters
        self.discount_factor = 0.99
        self.update_factor = 0.01
        self.replay_batch_size = 100
        self.P_optimizer, self.P_regularization, self.P_train_frequency = ('Adam', 0.01), ('L2', 0.01), 1
        self.Q_optimizer, self.Q_regularization, self.Q_train_frequency = ('Adam', 0.01), ('L2', 0.01), 1
        self.noise_lvl, self.noise_decay, self.noise_min = 1.0, 0.995, 0.05
        self.replay_buffer_init = 100
        self.train_on_episode_probability = 0.0

        # Replay buffer
        self.__rb_max_size = replay_buffer_size                                         # Replay buffer maximum size
        self.__rb_entries = 0                                                           # Replay buffer occupied entries
        self.__rb_obs = np.zeros((replay_buffer_size,) + environment.obs_sp_shape)      # Observed state buffer
        self.__rb_act = np.zeros((replay_buffer_size,) + environment.act_sp_shape)      # Action buffer
        self.__rb_nobs = np.zeros((replay_buffer_size,) + environment.obs_sp_shape)     # Next observed state buffer
        self.__rb_rwd = np.zeros((replay_buffer_size, 1))                               # Reward buffer
        self.__rb_end = np.zeros((replay_buffer_size, 1))                               # End buffer

        # Create the required directories if necessary
        if not os.path.isdir("./{0:s}".format(environment.name)):
            if os.path.isfile("./{0:s}".format(environment.name)):
                input("File './{0:s}' needs to be deleted. Press enter to continue.".format(environment.name))
                os.remove("./{0:s}".format(environment.name))
            os.mkdir("./{0:s}".format(environment.name))
            os.chdir("./{0:s}".format(environment.name))
            os.mkdir("./Init")
            os.mkdir("./TrainDDPG")
            os.mkdir("./TrainTD3")
            os.mkdir("./TrainSAC")
            os.mkdir("./TrainSAC2")
            with open('./TrainDDPG/Progress', 'wb') as file: np.save(file, 0)
            with open('./TrainTD3/Progress', 'wb') as file: np.save(file, 0)
            with open('./TrainSAC/Progress', 'wb') as file: np.save(file, 0)
            with open('./TrainSAC2/Progress', 'wb') as file: np.save(file, 0)
        else:
            os.chdir("./{0:s}".format(environment.name))

    @classmethod
    def DeepDeterministicPolicyGradient(cls, environment, policy_estimator, Q_function_estimator, replay_buffer_size):
        # Generic constructor
        learning_process = cls("Deep Deterministic Policy Gradient", environment, replay_buffer_size)

        # Policy estimators
        learning_process.Det_policy = policy_estimator
        learning_process.Sto_policy = policy_estimator

        # Action-Value function estimator
        learning_process.Q_function = Q_function_estimator

        # Value function estimator
        V_function = NeuralNetwork.connect("Value Function", policy_estimator, Q_function_estimator)
        V_function.freeze_layers([layer.name for layer in Q_function_estimator.layers])
        learning_process.V_function = V_function
        learning_process.V_target = NeuralNetwork.copy("Value Function Target", V_function)

        # Loss functions used to train the estimators
        learning_process.__Q_loss = "MeanSquaredError"
        learning_process.__V_loss = "Maximum"
        learning_process.__folder = "DDPG"

        # Return the object
        return learning_process

    @classmethod
    def TwinDelayedDeepDeterministicPolicyGradient(cls, environment, policy_estimator, Q_function_estimator, replay_buffer_size):
        # Generic constructor
        learning_process = cls("Twin Delayed Deep Deterministic Policy Gradient", environment, replay_buffer_size)

        # Policy estimators
        learning_process.Det_policy = policy_estimator
        learning_process.Sto_policy = policy_estimator

        # Action-Value function estimator
        Q2_function_estimator = NeuralNetwork.copy("", Q_function_estimator)
        for layer in Q2_function_estimator.layers: layer.name += " twin"
        input_name = Q_function_estimator.output_layers[0].name
        I = [NNLayer("Input", 1, name=input_name), NNLayer("Input", 1, name=input_name+" twin")]
        twin_Q_function = NeuralNetwork("", I, NNLayer("Minimum", 1, name="Minimum Q")(I))
        twin_Q_function = NeuralNetwork.connect("", Q_function_estimator, twin_Q_function)
        twin_Q_function = NeuralNetwork.connect("Twin action-value function", Q2_function_estimator, twin_Q_function)
        learning_process.Q_function = twin_Q_function

        # Value function estimator
        V_function_estimator = NeuralNetwork.connect("Value Function", policy_estimator, twin_Q_function)
        V_function_estimator.freeze_layers([layer.name for layer in twin_Q_function.layers])
        learning_process.V_function = V_function_estimator
        learning_process.V_target = NeuralNetwork.copy("Value Function Target", V_function_estimator)

        # Loss functions used to train the estimators
        learning_process.__Q_loss = "MeanSquaredError"
        learning_process.__V_loss = "Maximum"
        learning_process.__folder = "TD3"

        # Return the object
        return learning_process

    @classmethod
    def SoftActorCritic(cls, environment, policy_estimator, Q_function_estimator, replay_buffer_size, alpha):
        # Generic constructor
        learning_process = cls("Soft Actor Critic", environment, replay_buffer_size)

        # Policy estimators
        learning_process.Det_policy = policy_estimator
        policy_estimator = NeuralNetwork.copy("", policy_estimator, share_weights=True)
        out_layer = policy_estimator.output_layers[0]
        policy_estimator.output_layers.append(out_layer.input_layers[0])
        I = [NNLayer("Input", out_layer.in_shape, name=out_layer.input_layers[0].name),
             NNLayer("Input", out_layer.out_shape, name="Mean")]
        O = NNLayer("Exponential", out_layer.out_shape, name="Std. Dev.")(I[0])
        O = NNLayer("Gaussian", out_layer.out_shape, name=out_layer.name)([I[1], O])
        out_layer.name = "Mean"
        policy_estimator = NeuralNetwork.connect("Stochastic Policy", policy_estimator, NeuralNetwork("", I, O))
        learning_process.Sto_policy = policy_estimator

        # Action-Value function estimator
        Q2_function_estimator = NeuralNetwork.copy("", Q_function_estimator)
        for layer in Q2_function_estimator.layers: layer.name += " twin"
        input_name = Q_function_estimator.output_layers[0].name
        I = [NNLayer("Input", 1, name=input_name), NNLayer("Input", 1, name=input_name+" twin")]
        twin_Q_function = NeuralNetwork("", I, NNLayer("Minimum", 1, name="Minimum Q")(I))
        twin_Q_function = NeuralNetwork.connect("", Q_function_estimator, twin_Q_function)
        twin_Q_function = NeuralNetwork.connect("Twin action-value function", Q2_function_estimator, twin_Q_function)
        learning_process.Q_function = twin_Q_function

        # Value function estimator
        I = [NNLayer("Input", twin_Q_function.output_layers[0].out_shape, name=twin_Q_function.output_layers[0].name),
             NNLayer("Input", policy_estimator.output_layers[0].out_shape, name=policy_estimator.output_layers[0].name)]
        V_function_estimator = NeuralNetwork("", I, NNLayer("Entropy", 1, name="Twin Q function + Entropy")(I))
        V_function_estimator = NeuralNetwork.connect("", twin_Q_function, V_function_estimator)
        V_function_estimator = NeuralNetwork.connect("Value Function", policy_estimator, V_function_estimator)
        V_function_estimator.freeze_layers([layer.name for layer in twin_Q_function.layers])
        learning_process.V_function = V_function_estimator
        learning_process.V_target = NeuralNetwork.copy("Value Function Target", V_function_estimator)
        for layer in learning_process.V_function.layers + learning_process.V_target.layers:
            if layer.name == "Twin Q function + Entropy": layer.z = alpha

        # Loss functions used to train the estimators
        learning_process.__Q_loss = "MeanSquaredError"
        learning_process.__V_loss = "Maximum"
        learning_process.__folder = "SAC"

        # Return the object
        return learning_process

    @classmethod
    def SoftActorCritic2(cls, environment, policy_estimator, Q_function_estimator, replay_buffer_size):
        learning_process = cls.SoftActorCritic(environment, policy_estimator, Q_function_estimator, replay_buffer_size, 0.1)
        learning_process.__folder = "SAC2"
        learning_process.__alpha = 0.1
        return learning_process


    def __sample_replay_buffer(self, replay_size):
        '''
        Randomly selects and returns replay_size entries from the replay buffer
        :param replay_size: Number of entries to return (if greater than the current number of entries, all entries are returned)
        :return: Numpy arrays for the observed state, action, next observed state, reward and termination condition
        '''
        # Obtain indexes for replay_batch_size samples and return the corresponding entries (without duplicates)
        idx = list(np.random.choice(range(0, self.__rb_entries), size=min(replay_size, self.__rb_entries), replace=False))
        return self.__rb_obs[idx], self.__rb_act[idx], self.__rb_nobs[idx], self.__rb_rwd[idx], self.__rb_end[idx]
        # print("Index = ",idx," - Entries = ",self.__rb_entries, " / ",self.__rb_max_size," - Data = ",
        # (self.__rb_obs[idx], self.__rb_act[idx], self.__rb_nobs[idx], self.__rb_rwd[idx], self.__rb_end[idx]))

    def __store_to_replay_buffer(self, obs, act, next_obs, reward, end):
        '''
        Stores a new entry in the replay buffer (If the buffer is full overwrites a random entry)
        :param obs: Numpy array representing the observed state
        :param act: Numpy array representing the action
        :param next_obs: Numpy array representing the next observed state
        :param reward: Float representing the reward
        :param end: Boolean representing the termination condition
        :return:
        '''
        # Compute the next index and update the corresponding entry
        idx = self.__rb_entries if self.__rb_entries < self.__rb_max_size else np.random.randint(0, self.__rb_max_size)
        self.__rb_entries = min(self.__rb_entries + 1, self.__rb_max_size)
        self.__rb_obs[idx] = obs
        self.__rb_act[idx] = act
        self.__rb_nobs[idx] = next_obs
        self.__rb_rwd[idx] = reward
        self.__rb_end[idx] = end
        #print("Index = ",idx," - Entries = ",self.__rb_entries, " / ",self.__rb_max_size," - Data = ",
        # (obs, act, next_obs, reward, end))

    def train(self, episodes, ep_steps, resume_ep=-1, save_period=0, plot_period=0):
        '''
        Trains the model with the specified parameters
        :param episodes:          The number of episodes to run
        :param ep_steps:          The maximum number of steps in each episode
        :param replay_batch_size: The number of replay buffer's entries to use in training per executed step
        :param Q_tr_freq:         Number of steps between two training instances of the Q function estimator
        :param P_tr_freq:         Number of steps between two training instances of the policy estimator
        :param discount_factor:   Discount factor of the infinite-horizon discounted return
        :param update_factor:     Update factor of the target network
        :param act_noise:         Initial noise applied to the policy's action to allow exploration
        :param resume:            Flag indicating if the last training session should be resumed
        :param override:          Flag indicating if the last training session's parameters should replace the current ones
        :param save_period:       Number of episodes between two save operations of the algorithm's context
        :param plot_period:       Number of episodes between two plot operations
        :return:
        '''
        # Adjust parameters
        save_period = save_period if save_period else int(np.ceil(episodes/10))
        plot_period = plot_period if plot_period else episodes+1
        self.episodes, self.ep_steps = episodes, ep_steps

        # Retrieve the environment
        env = self.environment

        # Initialize algorithm variables
        ep_obs = np.zeros((ep_steps+1,) + env.obs_sp_shape)     # Episode's observed states
        ep_act = np.zeros((ep_steps,) + env.act_sp_shape)       # Episode's actions
        ep_rwd = np.zeros((ep_steps, 1))                        # Episode's rewards
        ep_ret = np.zeros((episodes, 3))                        # Returns for each episode (real, expected and RMSE)
        ep_loss = np.zeros((episodes, 2))                       # Training loss for each episode (Q function and Policy)
        self.ep_ret, self.ep_loss = ep_ret, ep_loss

        # Save/Load the training configuration
        if resume_ep == -1:
            # Initialize replay buffer if necessary
            if self.replay_buffer_init: self.__initialize_replay_buffer()

            # Reset the progress tracker and save the initial configuration
            episode = 0
            with open('./Train{0:s}/Progress'.format(self.__folder), 'wb') as file: np.save(file, 0)
            self.__save(episode)
        elif resume_ep == 1:
            # Obtain and load the last algorithm's context
            with open('./Train{0:s}/Progress'.format(self.__folder), 'rb') as file: episode = np.load(file)
            self.__load(episode)
        else:
            # Set the progress tracker and load the initial configuration
            episode = resume_ep
            with open('./Train{0:s}/Progress'.format(self.__folder), 'wb') as file: np.save(file, episode)
            self.__load(episode)

        # Retrieve the functions estimators
        P, Q, V, V_target = self.Sto_policy, self.Q_function, self.V_function, self.V_target

        # Retrieve the algorithm's hyper-parameters
        discount_factor, update_factor = self.discount_factor, self.update_factor
        replay_batch_size = self.replay_batch_size
        Q_loss, Q_opt, Q_reg, Q_freq = self.__Q_loss, self.Q_optimizer, self.Q_regularization, self.Q_train_frequency
        V_loss, V_opt, V_reg, V_freq = self.__V_loss, self.P_optimizer, self.P_regularization, self.P_train_frequency
        noise_lvl, noise_decay, noise_min = self.noise_lvl, self.noise_decay, self.noise_min
        tr_on_ep_prob = self.train_on_episode_probability

        for layer in self.V_function.layers:
            if layer.name == "Twin Q function + Entropy": entropy_layer = layer
        for layer in self.V_target.layers:
            if layer.name == "Twin Q function + Entropy": target_entropy_layer = layer
        alpha_aux0, alpha_aux1 = 0.0, 0.0
        beta1, beta2 = 0.9, 0.999
        beta1_aux, beta2_aux = 1.0, 1.0

        # Create and initialize plotter if enabled
        plotter = None
        if plot_period < episodes:
            plotter = Plotter(self.name+' - '+env.name, path='./Train{0:s}/'.format(self.__folder), figsize=(16, 9), resolution=50)
            plotter.initialize()
            plotter.update_all(episode)

        # For all episodes
        while episode < episodes:
            # Initialization
            ep_obs[0], s, end = env.reset(), 0, False

            # Do N steps following the policy estimator
            for s in range(ep_steps):
                ep_act[s] = P.compute(ep_obs[s])                               # Decide the next action
                ep_act[s] = np.clip(np.random.normal(ep_act[s], noise_lvl), -1.0, 1.0)          # Add clipped random noise
                ep_obs[s+1], ep_rwd[s], end = env.act(ep_act[s])                                # Act in the environment
#                ep_obs[s+1], ep_rwd[s], end = env.act(env.best_act())
                self.__store_to_replay_buffer(ep_obs[s], ep_act[s], ep_obs[s+1], ep_rwd[s], end)# Store in replay buffer
                if end: break                                                                   # End on terminal state
            ep_len = s + 1

            # Reduce noise
            noise_lvl = np.max([noise_decay * noise_lvl, noise_min])

            # Compute the real and expected returns and the root mean square error
            if not end: ep_rwd[s] += discount_factor * Q.compute([ep_obs[s+1], P.compute(ep_obs[s+1])])[0]
            for i in range(ep_len-2, -1, -1): ep_rwd[i] = ep_rwd[i] + discount_factor * ep_rwd[i + 1]
            ep_ret[episode, 0] = ep_rwd[0]
            ep_ret[episode, 1] = Q.compute([ep_obs[0], ep_act[0]])[0]
            ep_ret[episode, 2] = np.sqrt(np.square(ep_ret[episode, 0] - ep_ret[episode, 1]))

            for i in range(int(np.ceil(ep_len/Q_freq))):
                # Sample the replay buffer
                tr_obs, tr_act, tr_next_obs, tr_reward, tr_end = self.__sample_replay_buffer(replay_batch_size)
                tr_reward += (1 - tr_end) * discount_factor * V_target.compute(tr_next_obs)

                # Train Q
                ep_loss[episode,0] = np.mean(Q.train([tr_obs, tr_act], tr_reward, Q_loss, Q_opt, Q_reg, 1, 1))

                if i % V_freq == 0:
                    # Train V
                    ep_loss[episode,1] = np.mean(V.train(tr_obs, tr_reward, V_loss, V_opt, V_reg, 1, 1))

                    # Update target model's weights
                    V_target.update(V, update_factor)

                    if self.__alpha is not None:
                        u, s, r = entropy_layer.input_layers[1].input_layers[0].a, entropy_layer.input_layers[1].input_layers[1].a, entropy_layer.input_layers[1].z
                        a = u + r * s
                        h = np.mean(np.sum(0.5 * np.square(r) + np.log(np.sqrt(2 * np.pi) * s) - 2 * (a + np.log(0.5 + 0.5 * np.exp(-2 * a))), axis=-1, keepdims=True))

#                        entropy_layer.z -= 0.001 * (h + env.act_sp_shape[0])
                        beta1_aux *= beta1
                        beta2_aux *= beta2
                        alpha_aux0 = beta1 * alpha_aux0 + (1 - beta1) * (h + env.act_sp_shape[0])
                        alpha_aux1 = beta2 * alpha_aux1 + (1 - beta2) * np.square(h + env.act_sp_shape[0])
#                        entropy_layer.z -= (0.001 / (1 - beta1_aux)) * (alpha_aux0 / (np.square(alpha_aux1 / (1 - beta2_aux)) + 1E-8))
                        entropy_layer.z -= (0.001 / (1 - beta1_aux)) * (alpha_aux0 / (np.sqrt(alpha_aux1 / (1 - beta2_aux)) + 1E-8))
                        target_entropy_layer.z = entropy_layer.z
#                        ep_loss[episode, 1] = entropy_layer.z

            # Increase the episode number
            episode += 1

            # Save the algorithm's context and update the tracker file
            if (episode % save_period) == 0:
                self.__save(episode)
                with open('./Train{0:s}/Progress'.format(self.__folder), 'wb') as file: np.save(file, episode)

            if (episode % plot_period) == 0:
                plotter.update_trajectory(episode, ep_len, env.dest_pos, ep_obs[:ep_len+1, env.pos_idx], noise_lvl)
                if (episode % save_period) == 0: plotter.update_all(episode)

        if plotter: plotter.terminate()


    def __initialize_replay_buffer(self):
        ''' Initializes the replay buffer with the number of episodes specified in self.replay_buffer_init '''
        env = self.environment
        ep_steps = 1000
        ep_obs = np.zeros((ep_steps+1,) + env.obs_sp_shape)     # Episode's observed states
        ep_act = np.zeros((ep_steps,) + env.act_sp_shape)       # Episode's actions
        ep_rwd = np.zeros((ep_steps, 1))                        # Episode's rewards
        data = []                                               # List of lists of ep_len, ep_obs, ep_act and ep_reward
        episode, end = 0, False

        # Check if a saved file exist
        if os.path.isfile("./Init/replay_buffer.rbdf"):
            # Open it and load all available/needed episodes
            with open("./Init/replay_buffer.rbdf", 'rb') as file:
                for episode in range(np.min([np.load(file), self.replay_buffer_init])):
                    data.append([np.load(file), np.load(file), np.load(file), np.load(file)])
                episode += 1

        # Perform the missing episodes
        while episode < self.replay_buffer_init:
            # Reset the environment and perform N random steps
            ep_obs[0], s = env.reset(), 0
            for s in range(self.ep_steps):
                ep_act[s] = 2 * np.random.random_sample((1,) + env.act_sp_shape) - 1    # Select a random action
                ep_obs[s+1], ep_rwd[s], end = env.act(ep_act[s])                        # Act in the environment
                if end: break                                                           # End episode on terminal state

            # If the destination was reached, store it and increment the episode number
            if end and (ep_rwd[s] > 0):
                data.append([s+1, ep_obs[0:s+2].copy(), ep_act[0:s+1].copy(), ep_rwd[0:s+1].copy()])
                episode += 1
                print("\rReplay Buffer Init {0:.0f}%".format(np.floor((100.0*episode)/self.replay_buffer_init)), end='')

        # If there is at least one new episode, overwrite the saved file
        if end:
            with open("./Init/replay_buffer.rbdf", 'wb') as file:
                np.save(file, self.replay_buffer_init)      # Number of episodes
                for ep_data in data:
                    np.save(file, ep_data[0])               # Episode's length
                    np.save(file, ep_data[1])               # Episode's observed states
                    np.save(file, ep_data[2])               # Episode's actions
                    np.save(file, ep_data[3])               # Episode's rewards

        # Load the episodes' data in the replay buffer
        for e in range(self.replay_buffer_init):
            s = -1
            for s in range(data[e][0] - 1):
                self.__store_to_replay_buffer(data[e][1][s], data[e][2][s], data[e][1][s+1], data[e][3][s], False)
            self.__store_to_replay_buffer(data[e][1][s+1], data[e][2][s+1], data[e][1][s+2], data[e][3][s+1], True)
        return

    def __save(self, episode):
        ''' Saves the model's variables to a set of files '''
        # Get the file name
        filename = 'Train{0:s}/episode_{1:07d}'.format(self.__folder, episode)

        # Helper function
        def save_regularization(file, regularization):
            if regularization is None:
                np.save(file, 0)
            else:
                if not isinstance(regularization[0], (list, tuple)):
                    regularization = [regularization]
                np.save(file, len(regularization))
                for reg in regularization:
                    np.save(file, reg[0])
                    np.save(file, reg[1])

        # Save the current estimators
        self.Sto_policy.save('{0:s}.pnet'.format(filename))
        self.Q_function.save('{0:s}.qnet'.format(filename))
        self.V_target.save('{0:s}.tnet'.format(filename))

        # Save the algorithm's internal variables
        with open('{0:s}.rlcd'.format(filename), 'wb') as file:
            # Store the execution data
            np.save(file, self.episodes)
            np.save(file, self.ep_steps)
            np.save(file, self.__Q_loss)
            np.save(file, self.__V_loss)
            np.save(file, self.ep_ret[0:episode])
            np.save(file, self.ep_loss[0:episode])
            np.save(file, self.__rb_max_size)
            np.save(file, self.__rb_entries)
            np.save(file, self.__rb_obs[0:self.__rb_entries])
            np.save(file, self.__rb_act[0:self.__rb_entries])
            np.save(file, self.__rb_nobs[0:self.__rb_entries])
            np.save(file, self.__rb_rwd[0:self.__rb_entries])
            np.save(file, self.__rb_end[0:self.__rb_entries])

            # Store the hyper-parameters
            np.save(file, self.discount_factor)
            np.save(file, self.update_factor)
            np.save(file, self.replay_batch_size)
            np.save(file, self.Q_optimizer[0])
            np.save(file, self.Q_optimizer[1])
            save_regularization(file, self.Q_regularization)
            np.save(file, self.Q_train_frequency)
            np.save(file, self.P_optimizer[0])
            np.save(file, self.P_optimizer[1])
            save_regularization(file, self.P_regularization)
            np.save(file, self.P_train_frequency)
            np.save(file, self.noise_lvl)
            np.save(file, self.noise_decay)
            np.save(file, self.noise_min)
            np.save(file, self.replay_buffer_init)
            np.save(file, self.train_on_episode_probability)

            # Generate, set and store a random seed
            seed = np.random.randint(9999999, size=1)
            np.random.seed(seed)
            np.save(file, seed)

    def __load(self, episode):
        ''' Loads the model's variables from a set of files '''
        filename = 'Train{0:s}/episode_{1:07d}'.format(self.__folder, episode)

        # Helper function
        def load_regularization(file):
            regularization = []
            for i in range(np.load(file)):
                regularization.append((str(np.load(file)), np.load(file)))
            return None if regularization == [] else regularization

        # Load and set the estimators' weights
        self.Sto_policy.update(NeuralNetwork.load('{0:s}.pnet'.format(filename)), 1.0)
        self.Q_function.update(NeuralNetwork.load('{0:s}.qnet'.format(filename)), 1.0)
        self.V_target.update(NeuralNetwork.load('{0:s}.tnet'.format(filename)), 1.0)

        # Load the algorithm's internal variables
        with open('{0:s}.rlcd'.format(filename), 'rb') as file:
            # Recover the execution data
            self.episodes = np.load(file)
            self.ep_steps = np.load(file)
            self.__Q_loss = str(np.load(file))
            self.__V_loss = str(np.load(file))
            self.ep_ret[0:episode] = np.load(file)
            self.ep_loss[0:episode] = np.load(file)
            self.__rb_max_size = int(np.load(file))
            self.__rb_entries = int(np.load(file))
            self.__rb_obs[0:self.__rb_entries] = np.load(file)
            self.__rb_act[0:self.__rb_entries] = np.load(file)
            self.__rb_nobs[0:self.__rb_entries] = np.load(file)
            self.__rb_rwd[0:self.__rb_entries] = np.load(file)
            self.__rb_end[0:self.__rb_entries] = np.load(file)

            # Load the hyper-parameters
            self.discount_factor = np.load(file)
            self.update_factor = np.load(file)
            self.replay_batch_size = np.load(file)
            self.Q_optimizer = (str(np.load(file)), np.load(file))
            self.Q_regularization = load_regularization(file)
            self.Q_train_frequency = np.load(file)
            self.P_optimizer = (str(np.load(file)), np.load(file))
            self.P_regularization = load_regularization(file)
            self.P_train_frequency = np.load(file)
            self.noise_lvl = np.load(file)
            self.noise_decay = np.load(file)
            self.noise_min = np.load(file)
            self.replay_buffer_init = np.load(file)
            self.train_on_episode_probability = np.load(file)

            print("\nEpisodes", self.episodes, "\nSteps", self.ep_steps, "\nReplay Buffer", self.__rb_max_size,
                  "\nGamma", self.discount_factor, "\nTao", self.update_factor, "\nReplay Batch", self.replay_batch_size,
                  "\nQ opt", self.Q_optimizer, "\nQ reg", self.Q_regularization, "\nQ freq", self.Q_train_frequency,
                  "\nP opt", self.P_optimizer, "\nP reg", self.P_regularization, "\nP freq", self.P_train_frequency,
                  "\nNoise: ", self.noise_lvl, self.noise_decay, self.noise_min, "Buffer init", self.replay_buffer_init)

            # Recover the random seed
            np.random.seed(np.load(file))

    def test(self, ep_steps):
        ''' Tests the model using the graphic user interface '''
        # Initialize algorithm variables
        env = self.environment
        ep_obs = np.zeros((ep_steps+1,) + env.obs_sp_shape)   # Episode's observed states

        # Obtain the last episode number
        with open('./Train{0:s}/Progress'.format(self.__folder), 'rb') as file: episode = np.load(file)

        # Retrieve the environment and the estimators
        P = self.Det_policy

        a = 0.55
        if 1:
            # Create and initialize plotter if enabled
            for pos in np.array([[a, 0.01], [a, -a], [0.0, -a], [-a, -a], [-a, 0.0], [-a-0.02, a], [0.0, a], [a+0.03, a]]):
                # Obtain the starting position
                ep_obs[0], s = env.set_pos(pos.reshape((1,2))), 0
    #            plotter.receive_position().reshape((1, 2))

                # Act
                for s in range(ep_steps):
                    act = P.compute(ep_obs[s].reshape((1,) + env.obs_sp_shape))     # Decide the next action
                    ep_obs[s+1], _, end = env.act(act)                              # Act in the environment
                    if end: break                                                   # End the episode on terminal state

        plotter = Plotter(self.name+' - '+env.name, path='./Train{0:s}/'.format(self.__folder), figsize=(16, 9), resolution=50)
        plotter.initialize()
        plotter.update_all(episode)
        plotter.update_trajectory(episode, ep_steps, env.dest_pos, ep_obs[:, env.pos_idx], 0)

        # Replay buffer initialization
        while True:
            # Obtain the starting position
            pos = plotter.receive_position().reshape((1,2))
            print(pos)
            if False: break;
            ep_obs[0], s = env.set_pos(pos), 0

            # Act
            for s in range(ep_steps):
                act = P.compute(ep_obs[s].reshape((1,) + env.obs_sp_shape))     # Decide the next action
                ep_obs[s+1], _, end = env.act(act)                              # Act in the environment
                if end: break                                                   # End the episode on terminal state

            plotter.update_trajectory(episode, s+1, env.dest_pos, ep_obs[:s+2, env.pos_idx], 0)

        plotter.terminate()



