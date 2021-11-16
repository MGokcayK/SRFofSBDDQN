import numpy as np
import os 

from gNet import model as gmodel
from gNet import neuralnetwork as NN
from gNet import layer as glayer
from gNet import optimizer as gopt

class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions, batch_size) -> None:
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.state_mem = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.action_mem = np.zeros((self.mem_size, self.n_actions), dtype = np.int8)
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size)        

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        actions = np.zeros(self.action_mem.shape[1])
        actions[action] = 1.0
        self.action_mem[index] = actions
        self.reward_mem[index] = reward
        self.terminal_mem[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, new_states, terminals

def build_model(lr, n_actions, input_shape):
    model = gmodel.Model()
    model.add(glayer.Flatten(input_shape=(input_shape,)))
    model.add(glayer.Dense(256,'relu'))
    model.add(glayer.Dense(256,'relu'))
    model.add(glayer.Dense(n_actions))

    net = NN.NeuralNetwork(model)
    opt = gopt.Adam(lr=lr)
    net.setup(loss_function='mse', optimizer=opt)
    return net
		

class Agent():
    def __init__(self, LR, gamma, n_actions, batch_size,
                input_shape, epsilon=1.0, epsilon_dec=0.95, epsilon_min=0.1, 
                mem_size=5000, replace_target=100, model_name='gNet_RL', test=False) -> None:
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.LR = LR
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_name = model_name
        self.replace_target = replace_target
        self.action_type = 'Random Action'

        self.memory = ReplayBuffer(mem_size, input_shape, self.n_actions, self.batch_size)

        self.q_eval = build_model(self.LR, self.n_actions, input_shape)
        self.q_target = build_model(self.LR, self.n_actions, input_shape)

        if not test:
            self.save_model()
            self.load_model(training=True)
        else:
            self.load_model(training=False)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            self.action_type = 'Random Action'
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions.value)
            self.action_type = 'Model Prediction Action'

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:        
            state, action, reward, new_state, done = self.memory.sample_buffer()

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indicies = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            q_target = q_pred.value.copy()

            max_actions = np.argmax(q_eval.value, axis=1)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indicies] = reward + \
                    self.gamma * q_next.value[batch_index, max_actions.astype(int)] * done 

            self.q_eval.train_one_batch(state, q_target, printing=['no-print'])

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

            if (self.memory.mem_cntr % self.replace_target) == 0:
                self.save_model()
                self.load_model(training=True)

    def save_model(self):
        self.q_eval.save_model(self.model_name, False)

    def load_model(self, training=False):
        if training:
            modelName = self.model_name
        else:
            modelName = "models/"+ self.model_name 

        assert os.path.exists(modelName+".npy"), "RANDOMIZED MODEL LOADED PLEASE MAKE SURE THE MODEL IS UNDER `models` FOLDER LOADED!"

        if not training:
            self.q_eval.load_model(modelName, False)
        else:
            self.q_target.load_model(modelName, False)

























