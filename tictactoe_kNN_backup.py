import numpy as np
from collections import Counter
from scipy.spatial import distance

# Convert board state to a numerical format
def board_to_numeric(board):
    mapping = {'': 0, 'X': 1, 'O': 2}
    return [mapping[cell] for cell in board]

# Example dataset: (board state, next move)
dataset = [
    (['X', 'O', 'X', 'O', 'X', 'O', '', '', ''], 6),
    (['O', 'X', 'O', 'X', 'O', 'X', '', '', ''], 6),
    (['X', 'X', 'O', 'O', 'X', '', 'O', '', ''], 5),
    (['X', 'O', 'X', 'O', 'X', 'O', 'X', '', 'O'], 7),
    # Add more examples for a comprehensive dataset
]

# Convert dataset to numerical format
numeric_dataset = [(board_to_numeric(data[0]), data[1]) for data in dataset]

def knn_predict(board, dataset, k=3):
    numeric_board = board_to_numeric(board)
    # Calculate the distance between the input board and all boards in the dataset
    dists = sorted([(distance.euclidean(numeric_board, data[0]), data[1]) for data in dataset], key=lambda x: x[0])
    
    # Ensure we have at least k neighbors
    if len(dists) < k:
        k = len(dists)
    
    nearest_neighbors = [move for _, move in dists[:k]]
    
    # Filter out moves that are not valid (i.e., those that target already occupied cells)
    valid_moves = [move for move in nearest_neighbors if board[move] == '']
    
    if not valid_moves:
        raise ValueError("No valid moves available for AI.")
    
    # Return the most common move among the valid nearest neighbors
    return Counter(valid_moves).most_common(1)[0][0]

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

def check_winner(board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != '':
            return board[condition[0]]
    return None

def is_board_full(board):
    return '' not in board

def play_game():
    board = [''] * 9
    current_player = 'X'
    
    while True:
        print_board(board)
        
        if current_player == 'X':
            try:
                move = int(input("Enter your move (0-8): "))
                if move < 0 or move > 8:
                    raise ValueError
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 8.")
                continue
        else:
            try:
                move = knn_predict(board, numeric_dataset)
                print(f"AI chose move {move}")
            except ValueError as e:
                print(e)
                break
        
        if board[move] == '':
            board[move] = current_player
            winner = check_winner(board)
            
            if winner:
                print_board(board)
                print(f"Player {winner} wins!")
                break
            elif is_board_full(board):
                print_board(board)
                print("It's a tie!")
                break
            
            current_player = 'O' if current_player == 'X' else 'X'
        else:
            print("Invalid move. The cell is already occupied. Try again.")

play_game()
