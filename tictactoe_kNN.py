import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Initialize the game board
def create_board():
    return np.zeros((3, 3), dtype=int)

# Display the game board
def display_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for row in board:
        print(" | ".join([symbols[cell] for cell in row]))
        print("-" * 5)

# Check if a move is valid
def is_valid_move(board, row, col):
    return 0 <= row < 3 and 0 <= col < 3 and board[row, col] == 0

# Place a move on the board
def make_move(board, row, col, player):
    if is_valid_move(board, row, col):
        board[row, col] = player
        return True
    return False

# Check for a win
def check_win(board, player):
    # Check rows, columns, and diagonals
    return (np.any(np.all(board == player, axis=0)) or
            np.any(np.all(board == player, axis=1)) or
            np.all(np.diag(board) == player) or
            np.all(np.diag(np.fliplr(board)) == player))

# Training data
X_train = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, -1, 0, 0, 0, 0, 0],
    [1, 1, 0, -1, 0, 0, 0, 0, -1],
    [1, 1, 0, -1, 1, 0, 0, 0, -1],
    [1, 1, 0, -1, 1, 0, 0, -1, -1],
]

# Best move indices
y_train = [
    4,
    2,
    2,
    2,
    6,
    8,
]

# Train the kNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Function to predict the best move using kNN
def predict_best_move(board, model):
    board_flat = board.flatten().reshape(1, -1)
    best_move_index = model.predict(board_flat)[0]
    return divmod(best_move_index, 3)

# Get user move with error handling
def get_user_move():
    while True:
        try:
            row, col = map(int, input("Enter your move (row and column e.g. 0 0): ").split())
            if 0 <= row < 3 and 0 <= col < 3:
                return row, col
            else:
                print("Invalid move. Please enter values between 0 and 2.")
        except ValueError:
            print("Invalid input. Please enter two integers separated by a space.")

# The main game loop
def tic_tac_toe():
    board = create_board()
    current_player = 1  # Player 1 is X, Player -1 is O

    while True:
        display_board(board)
        if current_player == 1:
            row, col = get_user_move()
        else:
            row, col = predict_best_move(board, knn)
            while not is_valid_move(board, row, col):
                # If the predicted move is invalid, randomly select a valid move
                empty_cells = list(zip(*np.where(board == 0)))
                if not empty_cells:
                    break
                row, col = empty_cells[0]

        if make_move(board, row, col, current_player):
            if check_win(board, current_player):
                display_board(board)
                print("Player", current_player, "wins!")
                break
            if np.all(board != 0):
                display_board(board)
                print("It's a draw!")
                break
            current_player = -current_player
        else:
            print("Invalid move. Try again.")

tic_tac_toe()
