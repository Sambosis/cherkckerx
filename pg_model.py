# pg_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import logging
from torch.optim.lr_scheduler import StepLR
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BOARD_SIZE = 8
INPUT_SIZE = BOARD_SIZE * BOARD_SIZE
OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096 possible moves
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 256
LEARNING_RATE = 1e-3
GAMMA = 0.95
ENTROPY_COEFF = 0.01  # Entropy regularization coefficient
CLIP_GRAD = 1.0  # Gradient clipping threshold
STEP_SIZE = 1000  # Learning rate scheduler step size
GAMMA_SCHEDULER = 0.95  # Learning rate scheduler gamma

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE1, hidden2=HIDDEN_SIZE2, output_size=OUTPUT_SIZE):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden1)
        self.fc4 = nn.Linear(hidden1, output_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)

class PGAgent:
    def __init__(self, player, board_size=BOARD_SIZE, learning_rate=LEARNING_RATE, gamma=GAMMA, entropy_coeff=ENTROPY_COEFF, step_size=STEP_SIZE, gamma_scheduler=GAMMA_SCHEDULER):
        self.player = player
        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size * board_size * board_size

        self.model = PolicyNetwork(input_size=self.input_size, output_size=self.output_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma_scheduler)
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        logger.info(f"Initialized PGAgent for player {self.player}")
        
    def get_action(self, state, valid_moves):
        """
        Selects an action using the policy network.
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(DEVICE)  # Shape: [1, 64]
        probs = self.model(state_tensor)  # Shape: [1, 4096]
        
        # Convert valid moves to indices
        move_indices = [self._move_to_index(move) for move in valid_moves]
        if not move_indices:
            logger.warning("No valid moves available for PGAgent.")
            return None
        
        # Extract probabilities of valid moves
        valid_probs = probs[0][move_indices]
        sum_valid_probs = valid_probs.sum()
        
        if sum_valid_probs.item() == 0:
            # Assign equal probability if the sum is zero
            valid_probs = torch.ones_like(valid_probs) / len(valid_probs)
            logger.warning("Valid probabilities sum to zero. Assigning equal probabilities to all valid moves.")
        else:
            # Renormalize the probabilities
            valid_probs = valid_probs / sum_valid_probs
        
        # Check for NaNs and handle
        if torch.isnan(valid_probs).any():
            logger.error("NaN detected in valid_probs. Assigning equal probabilities to all valid moves.")
            valid_probs = torch.ones_like(valid_probs) / len(valid_probs)
        
        # Create a categorical distribution and sample an action
        dist = Categorical(valid_probs)
        try:
            action_idx = dist.sample()
        except Exception as e:
            logger.error(f"Error sampling action: {e}. Assigning random action.")
            return random.choice(valid_moves)
        
        selected_move = valid_moves[action_idx.item()]
        selected_move_index = move_indices[action_idx.item()]
        
        # Store state and action for training
        self.states.append(state.flatten())
        self.actions.append(selected_move_index)
        
        logger.debug(f"PGAgent selected move {selected_move} with probability {valid_probs[action_idx].item():.4f}")
        
        return selected_move
    
    def _move_to_index(self, move):
        """
        Converts a move tuple to a unique index.
        """
        from_row, from_col, to_row, to_col = move
        return ((from_row * self.board_size + from_col) * self.board_size + to_row) * self.board_size + to_col
    
    def remember(self, reward):
        """
        Stores the received reward.
        """
        self.rewards.append(reward)
        logger.debug(f"PGAgent received reward: {reward}")
        
    def train_policy(self):
        """
        Trains the policy network using the collected states, actions, and rewards.
        """
        if not self.states:
            logger.debug("No experiences to train on.")
            return
        
        # Convert lists to NumPy arrays first for efficiency
        states_np = np.array(self.states)  # Shape: [N, 64]
        actions_np = np.array(self.actions)  # Shape: [N]
        rewards_np = np.array(self.rewards)  # Shape: [N]
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards_np):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = np.array(discounted_rewards)
        
        # Normalize rewards
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) + 1e-9  # Add epsilon to prevent division by zero
        discounted_rewards = (discounted_rewards - mean) / std
        
        # Convert to torch tensors
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(DEVICE)
        states = torch.FloatTensor(states_np).to(DEVICE)  # Shape: [N, 64]
        actions = torch.LongTensor(actions_np).to(DEVICE)  # Shape: [N]
        
        # Forward pass
        probs = self.model(states)  # Shape: [N, 4096]
        selected_probs = probs[torch.arange(len(actions)), actions]  # Shape: [N]
        
        # Calculate log probabilities
        log_probs = torch.log(selected_probs + 1e-9)  # Add epsilon to prevent log(0)
        
        # Calculate loss (REINFORCE with entropy regularization)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
        loss = -torch.sum(log_probs * discounted_rewards) - self.entropy_coeff * entropy
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()  # Update the learning rate
        
        logger.debug(f"PGAgent training completed. Loss: {loss.item():.4f}")
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
    def adjust_parameters(self):
        """
        Adjusts hyperparameters such as the learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 1.2
            logger.info(f"PGAgent learning rate adjusted to: {param_group['lr']}")
        
    def save_model(self, filepath):
        """
        Saves the policy network's state dictionary to a file.
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"PGAgent model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Loads the policy network's state dictionary from a file.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=DEVICE))
        logger.info(f"PGAgent model loaded from {filepath}")