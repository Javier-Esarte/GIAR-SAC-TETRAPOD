import numpy as np
from CoppeliaSocket import CoppeliaSocket


class Environment:
    def __init__(self, obs_sp_shape, act_sp_shape, dest_pos, act_sp_ub, act_sp_lb):
        '''
        Creates a 3D environment using CoppeliaSim, where an agent capable of choosing its joints' angles tries to find
        the requested destination.
        :param obs_sp_shape: Numpy array's shape of the observed state
        :param act_sp_shape: Numpy array's shape of the action
        :param dest_pos: Destination position that the agent should search
        :param act_sp_ub: Numpy array containing the upper bounds of each element in the action array
        :param act_sp_lb: Numpy array containing the lower bounds of each element in the action array
        '''
        self.name = "ComplexAgent"
        self.obs_sp_shape = obs_sp_shape                        # Observation space shape
        self.act_sp_shape = act_sp_shape                        # Action space shape
        self.dest_pos = np.array(dest_pos)                      # Agent's destination
        self.pos_idx = tuple(i for i in range(len(dest_pos)))   # Position's indexes in the observed state array
        self.__pos_size = len(dest_pos)                         # Position's size
        self.__act_sp_m = (act_sp_ub-act_sp_lb)/2               # Action space linear transformation slope
        self.__act_sp_b = (act_sp_ub+act_sp_lb)/2               # Action space linear transformation y-intercept
        self.__end_cond = 0.1                                   # End condition
        self.__obs = np.zeros((1,)+self.obs_sp_shape)           # Observed state
        self.__coppelia = CoppeliaSocket(obs_sp_shape[0])       # Socket to the simulated environment

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Generate a new random starting position
        pos = 2 * np.random.rand(self.__pos_size+1) - 1
        while np.sqrt(np.sum(np.square(pos[0:2]))) < self.__end_cond:
            pos = 2 * np.random.random_sample(self.__pos_size+1) - 1

        # Reset the simulation environment and obtain the new state
        self.__obs = self.__coppelia.reset(pos)
        return np.copy(self.__obs)

    def set_pos(self, pos):
        ''' Sets and returns a new observed state for the environment '''
        # Reset the simulation environment and obtain the new state
        self.__obs = self.__coppelia.reset(pos.reshape(-1))
        return np.copy(self.__obs)

    def get_pos(self):
        ''' Returns the current position of the agent in the environment '''
        # Return the position
        return self.__obs[0:self.__pos_size]

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the
        obtained reward and the termination condition status '''
        # Take the requested action in the simulation and obtain the new state
        next_obs = self.__coppelia.act(act.reshape(-1))
        # Compute the reward
        reward, end = self.__compute_reward_and_end(self.__obs.reshape(1,-1), next_obs.reshape(1,-1))
        # Update the observed state
        self.__obs = np.copy(next_obs)
        # Return the environment's next state, the obtained reward and the termination condition status
        return next_obs, reward, end

    def compute_reward(self, obs):
        reward, _ = self.__compute_reward_and_end(obs[0:-1], obs[1:])
        return reward

    def compute_hindsight(self, obs, next_obs):
        obs, next_obs = np.copy(obs), np.copy(next_obs)
        obs[:, 0:self.__pos_size] -= next_obs[:, 0:self.__pos_size]
        next_obs[:, 0:self.__pos_size] -= next_obs[:, 0:self.__pos_size]
        reward, end = self.__compute_reward_and_end(obs, next_obs)
        return obs, next_obs, reward, end

    def __compute_reward_and_end(self, obs, next_obs):
        dist_ini = np.sqrt(np.sum(np.square(obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:self.__pos_size]), axis=1, keepdims=True))
        reward, end = np.zeros((dist_ini.shape[0], 1)), np.zeros((dist_ini.shape[0], 1))
        for i in range(dist_fin.shape[0]):
            if dist_fin[i] <= self.__end_cond:
                reward[i], end[i] = 100*(dist_ini[i]), True
            else:
                reward[i], end[i] = 100*(dist_ini[i]-dist_fin[i]), False if dist_fin[i] <= 1.5 else True
        return reward, end


if __name__ == '__main__':
    from NeuralNetworks import NNLayer, NeuralNetwork
    from ReinforcementLearning import ReinforcementLearning

    # Get the environment
    env = Environment(obs_sp_shape=(18,), act_sp_shape=(12,), dest_pos=(0,0), act_sp_ub=np.pi/4, act_sp_lb=-np.pi/4)

    # Input Layers
    S = NNLayer("Input", env.obs_sp_shape[0], name="State")
    A = NNLayer("Input", env.act_sp_shape[0], name="Action")

    # Policy estimator
    neurons = [[64, 32], [4, 2]][0]
    X = NNLayer("Relu", neurons[0], name="Dense A1")(S)
    X = NNLayer("Relu", neurons[1], name="Dense A2")(X)
    X = NNLayer("Tanh", env.act_sp_shape, name="Action")(X)
    P = NeuralNetwork("Policy estimator", S, X)

    # Q function estimator
    neurons = [[128, 64, 32], [8, 4, 2]][0]
    X = NNLayer("Relu", neurons[0], name="Dense Q1")([S, A])
    X = NNLayer("Relu", neurons[1], name="Dense Q2")(X)
    X = NNLayer("Relu", neurons[2], name="Dense Q3")(X)
    X = NNLayer("Linear", 1, name="Action Value")(X)
    Q = NeuralNetwork("Q function estimator", [S, A], X)

    # Create the model
    #model = ReinforcementLearning.SoftActorCritic(env, P, Q, replay_buffer_size=1000000, alpha=0.1)
    model = ReinforcementLearning.SoftActorCritic2(env, P, Q, replay_buffer_size=1000000)

    # Set training hyper-parameters
    model.discount_factor = 0.95
    model.update_factor = 0.005
    model.replay_batch_size = 1000
    model.P_optimizer, model.P_regularization, model.P_train_frequency = ('Adam', 0.001), None, 1
    model.Q_optimizer, model.Q_regularization, model.Q_train_frequency = ('Adam', 0.001), None, 3
    model.noise_lvl, model.noise_decay, model.noise_min = 0.00001, 0.9995, 0.00001
    model.replay_buffer_init = 0.0
    model.train_on_episode_probability = 0.0

    # Set start episode (previous run data is lost)
    resume = -1

#    model.V_function.print()
#    model.Sto_policy.print(print_weights=True)

    # Start training
    model.train(episodes=30000, ep_steps=100, resume_ep=resume, save_period=5000, plot_period=20)

#    model.test(ep_steps=100)


    import os
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    figsize = (14, 5)

    #if os.path.isfile('{0:s}.lrnd'.format("complex_agent")): os.remove('{0:s}.lrnd'.format("complex_agent"))
    if not os.path.isfile('{0:s}.lrnd'.format("complex_agent")):
        # Load results
        with open('{0:s}.lrnd'.format("complex_agent"), 'rb') as file:
            resolution = np.load(file)
            len = np.load(file)
            obs = np.load(file)
            rwd = np.load(file)
            ret = np.load(file)
            print(len, ret)

    else:
        # Compute the starting position grid
        resolution = 50
        axis_1D = np.linspace(-1, 1, num=resolution)
        mx, my = np.meshgrid(axis_1D, axis_1D)
        mx = np.array(mx).reshape(-1, 1)
        my = np.array(my).reshape(-1, 1)
        axis_2D = np.concatenate((mx, my), axis=1)
#        axis_2D = np.array([[1.0, 0.0], [0.707, 0.707], [0.0, 1.0], [-0.707, 0.707], [-1.0, 0.0], [-0.707, -0.707], [0.0, -1.0], [0.707, -0.707]])
#        axis_2D = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        axis_2D = np.array([[0.707, 0.707], [-0.707, 0.707], [-0.707, -0.707], [0.707, -0.707]])

        # Act in the environment from the different initial positions of the grid
        max_steps = 100
        obs = np.zeros((resolution*resolution, max_steps+1,) + env.obs_sp_shape)
        rwd = np.zeros((resolution*resolution, max_steps, 1))
        end = False
        len = np.zeros((resolution*resolution, 1))
        ret = np.zeros((resolution*resolution, 1))
        for i in range(axis_2D.shape[0]):
            obs[i,0] = env.set_pos(np.concatenate((axis_2D[i],[0.0,])))
            s = 1
            for s in range(max_steps):
                act = P.compute(obs[i,s].reshape((1,) + env.obs_sp_shape))
                obs[i,s+1], rwd[i,s], end = env.act(act)
                if end: break
            len[i] = s + 1
#            print(len[i], obs[i,0], obs[i,int(len[i])])
            if not end:
                print("ERROR: insufficient number of steps")
                ret[i] = 0.0
            else:
                # Compute the return
                discount_factor = 0.95
                ret[i] = rwd[i, int(len[i]) - 1]
                for j in range(int(len[i]) - 2, -1, -1): ret[i] = rwd[i,j] + discount_factor * ret[i]

        i = 0
        print(obs[i, :int(len[i])]-obs[i+1, :int(len[i])], rwd[i, :int(len[i])]-rwd[i+1, :int(len[i])], ret[i], ret[i+1])

        # Save results
        with open('{0:s}.lrnd'.format("complex_agent"), 'wb') as file:
            print("saved")
            np.save(file, resolution)
            np.save(file, len)
            np.save(file, obs)
            np.save(file, rwd)
            np.save(file, ret)

    # Plot results
    axis_1D = np.linspace(-1, 1, num=resolution)
    mx, my = np.meshgrid(axis_1D, axis_1D)
    colormap = mpl.colors.ListedColormap(mpl.cm.winter(np.linspace(0, 1, 1000))[0:1000, :-1]*0.9)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(mx, my, ret.reshape((resolution, resolution)), cmap=colormap)
    plt.show()

