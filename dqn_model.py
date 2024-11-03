# dqn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BOARD_SIZE = 8
INPUT_SIZE = BOARD_SIZE * BOARD_SIZE
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 256
OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096 possible moves
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
MIN_REPLAY_SIZE = 1000
MAX_LEARNING_RATE = 1e-2  # To prevent excessive learning rates

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class DuelingDQN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE1, hidden2=HIDDEN_SIZE2, output_size=OUTPUT_SIZE):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        # Value stream
        self.value_fc = nn.Linear(hidden2, hidden2)
        self.value = nn.Linear(hidden2, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden2, hidden2)
        self.advantage = nn.Linear(hidden2, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        return value + (advantage - advantage.mean())

class ReplayMemory:
    def __init__(self, capacity=MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, player, board_size=BOARD_SIZE):
        self.player = player
        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size * board_size * board_size
        self.model = DuelingDQN(input_size=self.input_size, output_size=self.output_size).to(DEVICE)
        self.target_model = DuelingDQN(input_size=self.input_size, output_size=self.output_size).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory()
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.target_update = TARGET_UPDATE_FREQ
        self.update_counter = 0
        
    def get_action(self, state, valid_moves):
        """
        Epsilon-greedy action selection with action masking.
        """
        if random.random() < self.epsilon:
            action = random.choice(valid_moves) if valid_moves else None
            if action is not None:
                logger.debug(f"Random action selected: {action}")
            return action
        
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        state = state.flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
        
        move_indices = [self._move_to_index(move) for move in valid_moves]
        if not move_indices:
            logger.warning("No valid moves available.")
            return None
        
        valid_q_values = q_values[move_indices]
        best_move_idx = np.argmax(valid_q_values)
        chosen_move = valid_moves[best_move_idx]
        logger.debug(f"Best action selected: {chosen_move} with Q-value: {valid_q_values[best_move_idx]}")
        return chosen_move
            
    def _move_to_index(self, move):
        """
        Converts a move to a unique index.
        """
        from_row, from_col, to_row, to_col = move
        return ((from_row * self.board_size + from_col) * self.board_size + to_row) * self.board_size + to_col
    
    def remember(self, state, action, reward, next_state, done):
        """
        Stores a transition in replay memory.
        """
        self.memory.push(state, action, reward, next_state, done)
        logger.debug(f"Transition stored: {action}, Reward: {reward}, Done: {done}")
        
    def train(self):
        """
        Samples a batch and performs a training step using Double DQN.
        """
        if len(self.memory) < MIN_REPLAY_SIZE:
            logger.debug("Not enough samples in memory to train.")
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare batches
        states = torch.FloatTensor([s.flatten() for s in batch.state]).to(DEVICE)  # Shape: [N, 64]
        next_states = torch.FloatTensor([s.flatten() for s in batch.next_state]).to(DEVICE)  # Shape: [N, 64]
        actions = torch.LongTensor([self._move_to_index(a) for a in batch.action]).unsqueeze(1).to(DEVICE)  # Shape: [N, 1]
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)  # Shape: [N, 1]
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(DEVICE)  # Shape: [N, 1]
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Select the best next actions from the current model
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            # Evaluate these actions using the target model
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            # Compute the target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            logger.debug(f"Epsilon decayed to: {self.epsilon}")
        
        # Update the target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logger.info("Target network updated.")
        
        logger.debug(f"Training step completed. Loss: {loss.item()}")
        
    def adjust_parameters(self):
        """
        Adjusts epsilon and learning rate based on external triggers from main loop.
        This method is called when the agent is underperforming or outperforming.
        """
        # Increase exploration rate, but cap at 1.0
        new_epsilon = min(self.epsilon * 1.2, 1.0)
        if new_epsilon > self.epsilon:
            self.epsilon = new_epsilon
            logger.info(f"Epsilon increased to: {self.epsilon}")
        
        # Increase learning rate, but cap at MAX_LEARNING_RATE
        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * 1.5
            if new_lr <= MAX_LEARNING_RATE:
                param_group['lr'] = new_lr
                logger.info(f"Learning rate increased to: {param_group['lr']}")
            else:
                logger.info(f"Learning rate not increased beyond max limit: {MAX_LEARNING_RATE}")
    
    def save_model(self, filepath):
        """
        Saves the model parameters.
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Loads the model parameters.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=DEVICE))
        self.target_model.load_state_dict(self.model.state_dict())
        logger.info(f"Model loaded from {filepath}")