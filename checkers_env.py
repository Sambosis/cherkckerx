import numpy as np
import pygame

class CheckersEnv:
    def __init__(self):
        self.board_size = 8
        self.reset()
        
        # Initialize pygame for visualization
        pygame.init()
        self.square_size = 80
        self.width = self.height = self.board_size * self.square_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Checkers')
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.KING_COLOR = (255, 255, 0)  # Yellow crown for kings
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 1 represents player 1's pieces, -1 represents player 2's pieces
        # Initialize player 1's pieces
        for i in range(3):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    self.board[i][j] = 1
        
        # Initialize player 2's pieces
        for i in range(5, 8):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    self.board[i][j] = -1
                        
        self.current_player = 1
        return self._get_state()
        
    def _get_state(self):
        return self.board.copy()
        
    def _get_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = self.board[i][j]
                if piece == self.current_player or piece == 2 * self.current_player:
                    is_king = abs(piece) == 2
                    # Normal moves
                    moves = self._get_piece_moves(i, j, is_king)
                    valid_moves.extend(moves)
                    
                    # Jump moves
                    jumps = self._get_jump_moves(i, j, is_king)
                    valid_moves.extend(jumps)
        return valid_moves
        
    def _get_piece_moves(self, row, col, is_king):
        moves = []
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All four diagonals
        else:
            if self.current_player == 1:
                directions = [(1, -1), (1, 1)]  # Forward diagonals for player 1
            else:
                directions = [(-1, -1), (-1, 1)]  # Forward diagonals for player -1
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                if self.board[new_row][new_col] == 0:
                    moves.append((row, col, new_row, new_col))
        return moves
        
    def _get_jump_moves(self, row, col, is_king):
        jumps = []
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All four diagonals
        else:
            if self.current_player == 1:
                directions = [(1, -1), (1, 1)]  # Forward diagonals for player 1
            else:
                directions = [(-1, -1), (-1, 1)]  # Forward diagonals for player -1
        
        for dr, dc in directions:
            jump_row, jump_col = row + 2 * dr, col + 2 * dc
            middle_row, middle_col = row + dr, col + dc
            if (0 <= jump_row < self.board_size and 0 <= jump_col < self.board_size):
                # Check if there is an opponent's piece to jump over
                middle_piece = self.board[middle_row][middle_col]
                opponent = -self.current_player
                if is_king:
                    # Kings can jump over both regular opponent pieces and opponent kings
                    if middle_piece == opponent or middle_piece == 2 * opponent:
                        if self.board[jump_row][jump_col] == 0:
                            jumps.append((row, col, jump_row, jump_col))
                else:
                    if middle_piece == opponent:
                        if self.board[jump_row][jump_col] == 0:
                            jumps.append((row, col, jump_row, jump_col))
        return jumps
        
    def step(self, action):
        from_row, from_col, to_row, to_col = action
        
        piece = self.board[from_row][from_col]
        is_king = abs(piece) == 2
        
        # Move the piece
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = 0
        
        # Handle jumps (captures)
        if abs(to_row - from_row) == 2:
            middle_row = (from_row + to_row) // 2
            middle_col = (from_col + to_col) // 2
            self.board[middle_row][middle_col] = 0
                
        # Check for kings
        if self.current_player == 1 and to_row == self.board_size - 1 and not is_king:
            self.board[to_row][to_col] = 2  # Promote to King for player 1
        elif self.current_player == -1 and to_row == 0 and not is_king:
            self.board[to_row][to_col] = -2  # Promote to King for player 2
                
        # Check if game is over
        done = self._is_game_over()
        reward = 1 if done else 0
                
        # Switch players
        self.current_player *= -1
                
        return self._get_state(), reward, done
        
    def _is_game_over(self):
        # Check if any player has no pieces left
        has_player1 = np.any(self.board > 0)
        has_player2 = np.any(self.board < 0)
        return not (has_player1 and has_player2)
        
    def render(self):
        # Handle Pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                    
        self.screen.fill(self.WHITE)
        
        # Draw board
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.screen, self.BLACK,
                                  (col * self.square_size, row * self.square_size,
                                   self.square_size, self.square_size))
                        
        # Draw pieces
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row][col]
                if piece != 0:
                    if piece > 0:
                        color = self.RED
                    else:
                        color = self.BLUE
                    center = (col * self.square_size + self.square_size // 2,
                            row * self.square_size + self.square_size // 2)
                    pygame.draw.circle(self.screen, color, center, self.square_size // 2 - 10)
                        
                    # Draw crown for kings
                    if abs(piece) == 2:
                        crown_radius = self.square_size // 8
                        crown_center = center
                        pygame.draw.circle(self.screen, self.KING_COLOR, crown_center, crown_radius)
            
        pygame.display.flip()
        pygame.time.wait(50)  # Adjust delay as needed for visualization
            
    def close(self):
        pygame.quit()