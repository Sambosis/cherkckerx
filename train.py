# main.py

import pygame
import numpy as np
from checkers_env import CheckersEnv
from dqn_model import DQNAgent
from pg_model import PGAgent
import time

def main():
    env = CheckersEnv()
    dqn_agent = DQNAgent(player=1)
    pg_agent = PGAgent(player=-1)
    
    num_episodes = 100000
    dqn_wins = 0
    pg_wins = 0
    RENDER_EVERY = 100  # Adjust as needed for visualization frequency
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            if episode % 10 == 0:
                print(f"Episode {episode}")
            # For visualization every RENDER_EVERY episodes
            show_game = (episode + 1) % RENDER_EVERY == 0
            
            while not done:
                # DQN agent's turn
                if env.current_player == 1:
                    valid_moves = env._get_valid_moves()
                    if not valid_moves:
                        done = True
                        pg_wins += 1
                        break
                        
                    action = dqn_agent.get_action(state, valid_moves)
                    next_state, reward, done = env.step(action)
                    
                    if done:
                        reward = 1 if np.any(next_state > 0) else -1
                        if reward == 1:
                            dqn_wins += 1
                        else:
                            pg_wins += 1
                            
                    dqn_agent.remember(state, action, reward, next_state, done)
                    dqn_agent.train()
                
                # PG agent's turn
                else:
                    valid_moves = env._get_valid_moves()
                    if not valid_moves:
                        done = True
                        dqn_wins += 1
                        break
                        
                    action = pg_agent.get_action(state, valid_moves)
                    next_state, reward, done = env.step(action)
                    
                    if done:
                        reward = 1 if np.any(next_state < 0) else -1
                        if reward == 1:
                            pg_wins += 1
                        else:
                            dqn_wins += 1
                            
                    pg_agent.remember(reward)
                    
                state = next_state
                
                if show_game:
                    env.render()
                    time.sleep(0.1)
                    
            # Train PG agent at the end of episode
            pg_agent.train_policy()
            
            # Adjust parameters every 100 episodes
            if (episode + 1) % 100 == 0:
                if dqn_wins < pg_wins:
                    dqn_agent.adjust_parameters()
                else:
                    pg_agent.adjust_parameters()
                    
                print(f"Episode {episode + 1}")
                print(f"DQN Wins: {dqn_wins}, PG Wins: {pg_wins}")
                dqn_wins = 0
                pg_wins = 0
                
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    main()