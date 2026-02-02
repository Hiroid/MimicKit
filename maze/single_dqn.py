import os
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pympler import asizeof
from visdom import Visdom
from copy import deepcopy
import networkx as nx
#this is set for the printing of Q-matrices via console
torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

from tqdm.auto import tqdm, trange

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#suppress scientific notation in printouts
np.set_printoptions(suppress=True)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

init_color_hex: str = "#2f78b6" # hex color of init point (left top)
goal_color_hex: str = '#f0b226' # hex color of goal point (right bottom)
wall_color: list[float] = [0.7, 0.7, 0.7] # RGB color of walls

n_cell: int = 7 # how many original cells
grid_dim = n_cell * 2 - 1 # side length of the square gridworld, in the form of 2n-1
n_actions: int = 4 # how many actions are possible in each state
lava: bool=False #do we use lava states - i.e. accessible wall states - (True) or wall states (False)? (we now always use "False")
starting_state: int = 0
goal_state = grid_dim ** 2 - 1

# generation of maze
maze_gen: bool = False # generate wall_states? wall_state_dict {0: [], 1: [1], 2: [2], 3: [3], ... } consists of 
n_mazes: int = 1

# rewards in the gridworld (-0.1, 100, -5)
step_reward: float = -0.1 # for taking a step
goal_reward: float = 2. # for reaching the goal
wall_reward: float = -1. # for bumping into a wall

# Net settings
input_neurons: int = 2 # for network init
output_neurons = n_actions # modeling the Q(s, a) value by the output neurons w.r.t. number of action
concept_size: int = 10 # the concept vector size of CA-TS DQN
using_norm: bool = True
hidden_dims: list = [2048, 1024, 512, 256]
dim_str = "-".join(str(d) for d in hidden_dims)
q_s2a: bool = False # whether using Q(s) -> a: True or Q(s, a): False
q_str = "" if q_s2a else "_sa2q"

# training settings
batch_size: int = 256 # 0 indicates using all data in buffer as a batch
epsilon: float = 0.1 # greedy action policy
lr: float = 2e-4 # learning rate
gamma_bellman: float = 0.9999 # bellman equation
target_replace_steps: int = 0 # renew target_net by eval_net after how many iter times, 0 indicates directly using eval_net as target_net
memory_capacity: int = 10000 # number transitions stored 
upper_steps: int = 300 # truncated steps
train_dqn: bool = True
n_episode: int = 1000
n_test_episode: int = 20
load_ckpt: bool = False
all_action_one_state: bool = True
ckpt_file: str = f"agent_ckpt/ckpt_{grid_dim}x{grid_dim}_n{n_mazes}{q_str}_dim{dim_str}_lr{lr}_gamma{gamma_bellman}_bs{batch_size}_tr{target_replace_steps}.pt"
memory_file: str = f"agent_memory/memory_{grid_dim}x{grid_dim}_n{n_mazes}{q_str}_dim{dim_str}_lr{lr}_gamma{gamma_bellman}_bs{batch_size}_tr{target_replace_steps}.npy"


class SquareGridworld():
    """
    the class for the gridworlds

    n**2 states (n-by-n gridworld) -> n is the grid dimension

    The mapping from state to the grid is as follows:
    n(n-1)  n(n-1)+1  ...  n^2-1
    ...     ...       ...  ...
    n       n+1       ...  2n-1
    0       1         ...  n-1

    Actions 0, 1, 2, 3 correspond to right, up, left, down (always exactly one step)

    -Landing in the goal_state gets a reward of goal_reward and ends the episode
    -Bumping into wall states or the map border incurs a reward of wall_reward
    -In case of lava=False, the agent bounces back from walls, while in case of lava=True wall states are accessible (the outside boundary can never be crossed)
    -Each step additionally incurs a reward of step_reward
    """
    def __init__(self, goal_state: int, wall_states: 'list[int]', lava: bool = False):

        self.goal_state: int=goal_state
        self.wall_states: 'list[int]'=wall_states
        self.n_states: int = grid_dim**2
        self.lava: bool = lava


    def get_outcome(self, state: int, action: int)-> 'tuple[Optional[int], float]':
        '''
        given a state and an action, this returns the next state and the immediate reward for the action
        ---
        INPUT
        state: the state the agent is in, can not equal to the goal_state
        action: the action taken
        ---
        OUTPUT
        next_state - the next state
        reward - the reward for the action taken
        '''

        #get the next state before taking into account walls or outside boundary
        next_state_dict={0:state+1, 1:state+grid_dim, 2:state-1, 3:state-grid_dim}
        #for all actions, this dictionary stores a boolean indicating whether we cross the map border executing this action in our current state
        cross_boundary_dict={0:state % grid_dim == grid_dim-1, 1:state >= grid_dim*(grid_dim-1), 2:state % grid_dim ==0, 3:state<grid_dim}

        #case that we have lava states that the agent can walk through with large negative reward
        if self.lava:
            reward=step_reward
            next_state=next_state_dict[action]
            #bounce back from map border
            if cross_boundary_dict[action]:
                reward+=wall_reward
                next_state=state
            #entering or exiting a lava state gives negative reward, but we do not bounce back
            elif next_state in self.wall_states or state in self.wall_states:
                reward+=wall_reward

        #case that we have wall states that the agent bounces back from
        else:
            reward=step_reward
            next_state=next_state_dict[action]
            #bounce back from a wall or the map border?
            if next_state in self.wall_states or cross_boundary_dict[action]:
                next_state=state #bounce back
                reward+=wall_reward
            
            if next_state == self.goal_state:
                reward=goal_reward


        return int(next_state), reward


    def get_outcomes(self) -> 'dict[ tuple[int,float] , tuple[Optional[int], float] ]':
        '''
        returns a dictionary where for every possible combination of state and action we get the next state and
        corresponding immediate reward
        ---
        OUTPUT
        outcomes - the dictionary keyed by state-action combo, whose values are the next states and rewards
        '''
        outcomes = {(s, a): self.get_outcome(s,a) for s in range(self.n_states) for a in range(n_actions)}
        return outcomes



def state_int_to_tuple(state_int: Optional[int]) -> Optional[torch.tensor]:
    '''
    Gets state tuple representation (coordinates) from integer representation
    ---
    INPUT
    state_int - the state integer representation
    ---
    OUTPUT
    state - the state tuple representation
    '''
    if state_int==None:
        return None
    else:
        cval: float=(grid_dim-1)/2 #center (0,0) is in the middle of the grid
        sx,sy=state_int%grid_dim-cval, mt.floor(state_int/grid_dim)-cval
        state = torch.tensor([[sx,sy]],device=device)
        return state

def batch_state_int_to_tuple(batch_state_int: list[int]) -> Optional[torch.tensor]:
    '''
    Gets batch state tuple representation (coordinates) from integer representation
    ---
    INPUT
    state_int - the state integer representation
    ---
    OUTPUT
    state - the state tuple representation
    '''
    state_tensor_list = []
    for state_int in batch_state_int:
        state_tensor = state_int_to_tuple(state_int)
        state_tensor_list.append(state_tensor)
    return torch.cat(state_tensor_list, dim = 0)

def state_tuple_to_int(state: Optional[torch.tensor]) -> Optional[int]:
    '''
    Gets state integer representation from tuple representation (coordinates)
    ---
    INPUT
    state - the state tuple representation
    ---
    OUTPUT
    state_int - the state integer representation
    '''
    if state==None:
        return None
    else:
        cval: float = (grid_dim-1)/2
        sx = state[:, 0] + cval
        sy = state[:, 1] + cval
        state_int = sx + grid_dim * sy
        return state_int.tolist()


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.neighbours = []

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self._link_neighbours()

    def _link_neighbours(self):
        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                if x > 0:
                    cell.add_neighbour(self.cells[x - 1][y])
                if x < self.width - 1:
                    cell.add_neighbour(self.cells[x + 1][y])
                if y > 0:
                    cell.add_neighbour(self.cells[x][y - 1])
                if y < self.height - 1:
                    cell.add_neighbour(self.cells[x][y + 1])

    def random_cell(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return self.cells[x][y]

class Maze:
    def __init__(self, width, height):
        self.grid = Grid(width, height)
        self.width = width
        self.height = height
        self.maze_array = np.ones((2 * width + 1, 2 * height + 1), dtype=bool)  # 1 for walls, 0 for paths, including boundary
        self.states_array = np.ones((2 * width - 1, 2 * height - 1), dtype=bool)  # 1 for walls, 0 for paths, w/o boundary
        self.num_moved_walls = 0

    def mark_as_visited(self, cell):
        cell.visited = True
        self.maze_array[2 * cell.x + 1][2 * cell.y + 1] = 0  # Mark as path

    def create_passage(self, cell1, cell2):
        # Remove wall between two cells
        x1, y1 = 2 * cell1.x + 1, 2 * cell1.y + 1
        x2, y2 = 2 * cell2.x + 1, 2 * cell2.y + 1
        self.maze_array[(x1 + x2) // 2][(y1 + y2) // 2] = 0
        self.num_moved_walls += 1

    def generate(self):
        stack = []
        current_cell = self.grid.random_cell()
        self.mark_as_visited(current_cell)

        while True:
            unvisited_neighbours = [n for n in current_cell.neighbours if not n.visited]
            if unvisited_neighbours:
                next_cell = random.choice(unvisited_neighbours)
                self.create_passage(current_cell, next_cell)
                stack.append(current_cell)
                current_cell = next_cell
                self.mark_as_visited(current_cell)
            elif stack:
                current_cell = stack.pop()
            else:
                break
        
        self.states_array = self.maze_array[1:-1, 1:-1]

def plot_maze(states_array):
    wall_height, wall_weight = states_array.shape
    maze_array = np.ones((wall_height + 2, wall_weight + 2), dtype=bool)
    maze_array[1:-1, 1:-1] = states_array
    
    maze_height, maze_width = maze_array.shape

    rgb_image = np.ones((maze_height, maze_width, 3))
    rgb_image[maze_array == 1] = wall_color

    # init
    rgb_image[maze_height - 2, 1] = tuple(int(init_color_hex[i:i+2], 16) / 255 for i in (1, 3, 5))
    rgb_image[maze_height - 2, 0] = [1, 1, 1]

    # goal
    rgb_image[1, maze_width - 2] = tuple(int(goal_color_hex[i:i+2], 16) / 255 for i in (1, 3, 5))
    rgb_image[1, maze_width - 1] = [1, 1, 1]

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()

def maze_array2states(states_array: np.ndarray) -> list[int]:
    wall_states = []
    n_rows, n_cols = states_array.shape
    for i in range(n_rows - 1, -1, -1):
        for j in range(n_cols):
            if states_array[i, j]:
                pos_code = (n_rows - 1 - i) * n_cols + j
                wall_states.append(pos_code)

    return wall_states

def maze_states2array(wall_states: list[int]) -> np.ndarray:
    n = grid_dim
    states_array = np.full((n, n), False, dtype=bool)
    
    for pos_code in wall_states:
        j = pos_code % n
        i = n - 1 - (pos_code // n)
        states_array[i, j] = True
    
    return states_array


class Net_q_s2a(nn.Module):
    def __init__(self):
        super(Net_q_s2a, self).__init__()
        self.using_norm = using_norm

        self.ts_fc_layers = nn.ModuleDict()
        self.ts_norm_layers = nn.ModuleDict()

        prev_dim = input_neurons
        for i, dim in enumerate(hidden_dims):
            self.ts_fc_layers[f'fc{i}'] = nn.Linear(prev_dim, dim, bias=True)
            if self.using_norm:
                self.ts_norm_layers[f'norm{i}'] = nn.LayerNorm(dim)
            prev_dim = dim

        self.ts_fc_layers[f'fc{len(hidden_dims)}'] = nn.Linear(prev_dim, output_neurons, bias=True)

        self.cdp_fc_layers = nn.ModuleDict()
        self.cdp_norm_layers = nn.ModuleDict()

        prev_dim = concept_size
        for i, dim in enumerate(hidden_dims):
            self.cdp_fc_layers[f'fc{i}'] = nn.Linear(prev_dim, dim, bias=True)
            if self.using_norm:
                self.cdp_norm_layers[f'bn{i}'] = nn.LayerNorm(dim)
            prev_dim = dim

        self.ts_afun = nn.ReLU()
        self.cdp_afun = nn.Sigmoid()

        self.concept_embedding_layer = nn.Embedding(num_embeddings=n_mazes, embedding_dim=concept_size)

    def forward(self, x, concept_idx=None):
        if concept_idx is not None:
            concept = self.concept_embedding_layer(concept_idx)
            cdp_activations = []
            c = concept
            for i in range(len(self.cdp_fc_layers)):
                c = self.cdp_fc_layers[f'fc{i}'](c)
                if self.using_norm:
                    c = self.cdp_norm_layers[f'bn{i}'](c)
                c = self.cdp_afun(c)
                cdp_activations.append(c)

        for i in range(len(self.ts_fc_layers) - 1):
            x = self.ts_fc_layers[f'fc{i}'](x)
            if self.using_norm and f'bn{i}' in self.ts_norm_layers:
                x = self.ts_norm_layers[f'bn{i}'](x)
            x = self.ts_afun(x)
            if concept_idx is not None:
                x = torch.mul(x, cdp_activations[i])

        x = self.ts_fc_layers[f'fc{len(self.cdp_fc_layers)}'](x)

        return x

class Net_sa2q(nn.Module):
    def __init__(self):
        super(Net_sa2q, self).__init__()
        self.using_norm = using_norm

        self.ts_fc_layers = nn.ModuleDict()
        self.ts_norm_layers = nn.ModuleDict()

        prev_dim = input_neurons + n_actions # dim 0, 1 is xy coordinates, dim 2 to 5 is action from 0 to 3 (right, uo, left, down)
        for i, dim in enumerate(hidden_dims):
            self.ts_fc_layers[f'fc{i}'] = nn.Linear(prev_dim, dim, bias=True)
            if self.using_norm:
                self.ts_norm_layers[f'bn{i}'] = nn.LayerNorm(dim)
            prev_dim = dim

        self.ts_fc_layers[f'fc{len(hidden_dims)}'] = nn.Linear(prev_dim, 1, bias=True)

        self.cdp_fc_layers = nn.ModuleDict()
        self.cdp_norm_layers = nn.ModuleDict()

        prev_dim = concept_size
        for i, dim in enumerate(hidden_dims):
            self.cdp_fc_layers[f'fc{i}'] = nn.Linear(prev_dim, dim, bias=True)
            if self.using_norm:
                self.cdp_norm_layers[f'bn{i}'] = nn.LayerNorm(dim)
            prev_dim = dim

        self.ts_afun = nn.ReLU()
        self.cdp_afun = nn.Sigmoid()

        self.concept_embedding_layer = nn.Embedding(num_embeddings=n_mazes, embedding_dim=concept_size)

    def forward(self, x, concept_idx=None):
        if concept_idx is not None:
            concept = self.concept_embedding_layer(concept_idx)
            cdp_activations = []
            c = concept
            for i in range(len(self.cdp_fc_layers)):
                c = self.cdp_fc_layers[f'fc{i}'](c)
                if self.using_norm:
                    c = self.cdp_norm_layers[f'bn{i}'](c)
                c = self.cdp_afun(c)
                cdp_activations.append(c)

        for i in range(len(self.ts_fc_layers) - 1):
            x = self.ts_fc_layers[f'fc{i}'](x)
            if self.using_norm and f'bn{i}' in self.ts_norm_layers:
                x = self.ts_norm_layers[f'bn{i}'](x)
            x = self.ts_afun(x)
            if concept_idx is not None:
                x = torch.mul(x, cdp_activations[i])

        x = self.ts_fc_layers[f'fc{len(self.cdp_fc_layers)}'](x)

        return x

class DQN(object):
    def __init__(self, device):

        self.device = device
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.lr = lr
        self.gamma_bellman = gamma_bellman
        self.target_replace_steps = target_replace_steps
        self.memory_capacity = memory_capacity

        if q_s2a:
            self.eval_net, self.target_net = Net_q_s2a().to(self.device), Net_q_s2a().to(self.device)
        else:
            self.eval_net, self.target_net = Net_sa2q().to(self.device), Net_sa2q().to(self.device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, 4))
        self.optimizer_concept = torch.optim.Adam(self.eval_net.concept_embedding_layer.parameters(), lr = self.lr)
        other_params = [param for name, param in self.eval_net.named_parameters() if not name.startswith('concept_embedding_layer')]
        self.optimizer_net = torch.optim.Adam(other_params, lr = self.lr)
        
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, state_int):
        state_tensor = state_int_to_tuple(state_int) # tensor shape (1, 2)
        # input only one sample
        if np.random.uniform() > self.epsilon: # greedy
            self.eval_net.eval()
            
            if q_s2a:
                actions_value = self.eval_net(state_tensor)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()
                action = action[0]
            else:
                state_tensor = state_tensor.repeat_interleave(n_actions, dim=0)
                eye_tensor = torch.eye(n_actions, device = device)
                state_action_tensor = torch.cat((state_tensor, eye_tensor), dim = 1)
                q_state_all_actions = self.eval_net(state_action_tensor)
                action = torch.argmax(q_state_all_actions, dim = 0).item()

            self.eval_net.train()
        else:
            action = np.random.randint(0, n_actions)
        return action
    
    def store_transition(self, state_int, a, reward, next_state_int, no_repeat = False):
        transition = np.hstack((state_int, a, reward, next_state_int))
        
        if no_repeat:
            if not any(np.array_equal(transition, mem) for mem in self.memory):
                # replace the old memory with new memory
                index = self.memory_counter % self.memory_capacity
                self.memory[index, :] = transition
                self.memory_counter += 1
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
            self.memory_counter += 1
        

    def learn(self, train_net = True, train_concept = True):
        
        exec_batch_size = self.memory_counter if self.batch_size == 0 else self.batch_size
            
        # target parameter update
        if self.target_replace_steps != 0 and self.learn_step_counter % self.target_replace_steps == 0:
            self.target_net.load_state_dict(deepcopy(self.eval_net.state_dict()))
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(min(self.memory_capacity, self.memory_counter), exec_batch_size, replace = False)
        batch_memory = self.memory[sample_index, :]
        batch_state_int = batch_memory[:, 0].tolist()
        batch_a = torch.tensor(batch_memory[:, 1], dtype=torch.int64, device = device).view(-1, 1)
        batch_reward = torch.tensor(batch_memory[:, 2], dtype=torch.float32, device = device).view(-1, 1)
        batch_done = (batch_reward == goal_reward).float()
        batch_next_state_int = batch_memory[:, 3].tolist()

        batch_state_tensor = batch_state_int_to_tuple(batch_state_int).to(device) # shape (batch, 2)
        batch_next_state_tensor = batch_state_int_to_tuple(batch_next_state_int).to(device) # shape (batch, 2)

        if train_net:
            if q_s2a:
                q_eval = self.eval_net(batch_state_tensor).gather(1, batch_a) # shape (batch, 1)
                
                if self.target_replace_steps != 0:
                    q_next = self.target_net(batch_next_state_tensor).detach()
                else: 
                    q_next = self.eval_net(batch_next_state_tensor).detach()
                
            else:
                batch_action_tensor = F.one_hot(batch_a.squeeze(), num_classes = n_actions)
                batch_state_action_tensor = torch.cat((batch_state_tensor, batch_action_tensor.float()), dim = 1)
                q_eval = self.eval_net(batch_state_action_tensor)

                candidate_action_tensor = torch.eye(n_actions, device = device).unsqueeze(0).repeat(batch_next_state_tensor.size(0), 1, 1)
                candidate_next_state_tensor = batch_next_state_tensor.unsqueeze(1).repeat(1, n_actions, 1)
                candidate_state_action_tensor = torch.cat((candidate_next_state_tensor, candidate_action_tensor), dim = 2)

                if self.target_replace_steps != 0:
                    q_next = self.target_net(candidate_state_action_tensor.view(exec_batch_size * 4, -1)).detach().view(exec_batch_size, -1)
                else:
                    q_next = self.eval_net(candidate_state_action_tensor.view(exec_batch_size * 4, -1)).detach().view(exec_batch_size, -1)

                # print(
                #     f"batch_action_tensor shape {batch_action_tensor.shape}, expected (batch_size, 4) \n"
                #     f"batch_state_action_tensor shape {batch_state_action_tensor.shape}, expected (batch_size, 6) \n"
                #     f"q_eval shape {q_eval.shape}, expected (batch_size, 1) \n"
                #     f"candidate_action_tensor shape {candidate_action_tensor.shape}, expected (batch_size, 4, 4) \n"
                #     f"candidate_next_state_tensor shape {candidate_next_state_tensor.shape}, expected (batch_size, 4, 2) \n"
                #     f"candidate_state_action_tensor shape {candidate_state_action_tensor.shape}, expecetd (batch, 4, 6) \n"
                #     f"out shape {q_next.shape}, expected (batch, 4)"
                # )
                # raise ValueError("Dev mode")

            q_target = batch_reward + torch.mul(self.gamma_bellman * q_next.max(1)[0].view(exec_batch_size, 1), 1 - batch_done) # shape (batch, 1)

            loss = self.loss_func(q_eval, q_target)

            self.optimizer_net.zero_grad()
            loss.backward()
            self.optimizer_net.step()

        if train_concept:
            q_eval2 = self.eval_net(batch_state_tensor).gather(1, batch_a) # shape (batch, 1)
            q_next2 = self.target_net(batch_next_state_tensor).detach()
            q_target2 = batch_reward + self.gamma_bellman * q_next2.max(1)[0].view(exec_batch_size, 1) # shape (batch, 1)
            loss2 = self.loss_func(q_eval2, q_target2)

            self.optimizer_concept.zero_grad()
            loss2.backward()
            self.optimizer_concept.step()

        return loss.item()


def play(env, dqn):
    episode_durations = []
    ep_r_durations = []
    if load_ckpt:
        dqn.eval_net.load_state_dict(torch.load(ckpt_file))
        dqn.memory = np.load(memory_file)
        dqn.memory_counter = np.count_nonzero(dqn.memory[:, 2])
    print('\nCollecting experience...')

    try:
        for i_episode in range(n_episode):
            if i_episode >= n_episode / 2 or train_dqn == False:
                start_state_int = 0
            else:
                while True:
                    start_state_int = random.randint(starting_state, goal_state - 1)
                    if start_state_int not in env.wall_states:
                        break
                    
            state_int = start_state_int 
            ep_r = 0
            steps = 0
        
            while True:
                loss = 0
                done = False
                a = dqn.choose_action(state_int)
                next_state_int, reward = env.get_outcome(state_int, a)
                ep_r += reward
                steps += 1

                if steps == upper_steps or next_state_int == goal_state: done = True

                if train_dqn:
                    if all_action_one_state:
                        next_state_int_0, reward_0 = env.get_outcome(state_int, 0)
                        next_state_int_1, reward_1 = env.get_outcome(state_int, 1)
                        next_state_int_2, reward_2 = env.get_outcome(state_int, 2)
                        next_state_int_3, reward_3 = env.get_outcome(state_int, 3)

                        dqn.store_transition(state_int, 0, reward_0, next_state_int_0, no_repeat = True)
                        dqn.store_transition(state_int, 1, reward_1, next_state_int_1, no_repeat = True)
                        dqn.store_transition(state_int, 2, reward_2, next_state_int_2, no_repeat = True)
                        dqn.store_transition(state_int, 3, reward_3, next_state_int_3, no_repeat = True)
                    else:
                        dqn.store_transition(state_int, a, reward, next_state_int, no_repeat = True)
                    
                    if dqn.batch_size <= min(dqn.memory_capacity, dqn.memory_counter): 
                        loss = dqn.learn(train_net = True, train_concept = False)

                if done:
                    episode_durations.append(steps)
                    ep_r_durations.append(ep_r)
                    print(
                        f"Episode: {i_episode}, | "
                        f"Ep_r: {round(ep_r, 2)}, | "
                        f"loss: {loss:.6f}, | "
                        f"Steps: {steps}, | "
                        f"Memory counter: {dqn.memory_counter} | "
                        f"Start: {state_int_to_tuple(start_state_int).tolist()} | "
                        f"Goal: {state_int_to_tuple(goal_state).tolist()}"
                    )
                    break
                
                state_int = next_state_int

    except KeyboardInterrupt:
        pass
    
    for i_episode in range(n_test_episode):
        if i_episode >= n_test_episode / 2:
            start_state_int = 0
        else:
            while True:
                start_state_int = random.randint(starting_state, goal_state - 1)
                if start_state_int not in env.wall_states:
                    break

        state_int = start_state_int
        ep_r = 0
        steps = 0
    
        while True:
            done = False
            a = dqn.choose_action(state_int)
            next_state_int, reward = env.get_outcome(state_int, a)
            ep_r += reward
            steps += 1

            if steps == upper_steps or next_state_int == goal_state: done = True

            if done:
                episode_durations.append(steps)
                ep_r_durations.append(ep_r)
                print(
                    f"Test episode: {i_episode}, | "
                    f"Ep_r: {round(ep_r, 2)}, | "
                    f"Steps: {steps}, | "
                    f"Memory counter: {dqn.memory_counter} | "
                    f"Start: {state_int_to_tuple(start_state_int).tolist()} | "
                    f"Goal: {state_int_to_tuple(goal_state).tolist()}"
                )
                break
            
            state_int = next_state_int

    if train_dqn:    
        torch.save(dqn.eval_net.state_dict(), ckpt_file)
        np.save(memory_file, dqn.memory)
        print(f'Single DQN training done!')
    else:
        print('Evaluating Single DQN done!')
    print(f"mean steps and ep_r of last 10 steps is {np.mean(episode_durations[-10:])} and {np.mean(ep_r_durations[-10:])}")

if maze_gen:
    wall_array_dict = {}

    for i in range(n_mazes):
        maze = Maze(n_cell, n_cell)
        maze.generate()
        wall_array_dict[i] = maze.states_array
    
    torch.save(wall_array_dict, f"maze_wall_array/wall_array_dict_{grid_dim}x{grid_dim}_n{n_mazes}.pt")

else:
    wall_array_dict = torch.load(f"maze_wall_array/wall_array_dict_{grid_dim}x{grid_dim}_n{n_mazes}.pt")

print(f"{asizeof.asizeof(wall_array_dict)/1024/1024:.2f}M")

wall_state_dict = {}

for i in range(n_mazes):
    wall_state_dict[i] = maze_array2states(wall_array_dict[i])

set_seed(0)
env = SquareGridworld(goal_state = grid_dim**2 - 1, wall_states = wall_state_dict[0])
plot_maze(maze_states2array(wall_state_dict[0]))

print(
    f"using_norm: bool = {using_norm} \n"
    f"hidden_dims: list = {hidden_dims}\n"
    f"q_s2a: bool = {q_s2a} # whether using Q(s) -> a: True or Q(s, a): False\n"
    f"batch_size: int = {batch_size} # 0 indicates using all data in buffer as a batch\n"
    f"epsilon: float = {epsilon} # greedy action policy\n"
    f"lr: float = {lr} # learning rate\n"
    f"gamma_bellman: float = {gamma_bellman} # bellman equation\n"
    f"target_replace_steps: int = {target_replace_steps} # renew target_net by eval_net after how many iter times, 0 indicates directly using eval_net as target_net\n"
    f"memory_capacity: int = {memory_capacity} # number transitions stored \n"
    f"upper_steps: int = {upper_steps} # truncated steps\n"
    f"train_dqn: bool = {train_dqn}\n"
    f"n_episode: int = {n_episode}\n"
    f"n_test_episode: int = {n_test_episode}\n"
    f"load_ckpt: bool = {load_ckpt}\n"
    f"all_action_one_state: bool = {all_action_one_state}\n"
)
dqn = DQN(device)
play(env, dqn)
