import array
from typing import Tuple
import torch
import numpy as np
from torch.autograd import Variable

from mancala.game import copy_board, flip_board, play_turn

def sigmoid(net):
    return (2 / (1 + np.exp(-2 * net))) - 1

def sigmoid_prime(net):
    return 1 - sigmoid(net)**2

def evaluate_heuristics(B, player):
    # H1: Hoard as many seeds as possible in one pit.
    h1 = max(B[7:13]) if player == 1 else max(B[0:6])
    # H2: Keep as many seeds on the player's own side.
    h2 = sum(B[7:13]) if player == 1 else sum(B[0:6])
    # H3: Have as many moves as possible from which to choose.
    h3 = len(getLegalMoves(B, player))
    # H4: Maximise the amount of seeds in a player's own store.
    h4 = B[13] if player == 1 else B[6]
    # H5: Move the seeds from the pit closest to the opponent's side.
    # For Player 1, this is index 12, and for Player 0, it is index 5.
    h5 = B[12] if player == 1 else B[5]
    # H6: Keep the opponent's score to a minimum.
    # This heuristic requires looking ahead two moves, which is complex.
    # For simplicity, we'll use the current score of the opponent.
    h6 = -B[6] if player == 1 else -B[13]
    return [h1, h2, h3, h4, h5, h6]

def TD_0(board, player, alpha, gamma, num_episodes):
    weights = np.zeros(6)
    for i in range(num_episodes):
        state = copy_board(board)
        while True:
            # Evaluate the current state
            x = evaluate_heuristics(state, player)
            net = np.dot(weights, x)
            V = sigmoid(net)

            # Choose a move
            legal_moves = getLegalMoves(state, player)
            if len(legal_moves) == 0:
                break
            move = legal_moves[np.random.randint(len(legal_moves))]

            # Update the weights
            next_state = copy_board(state)
            play_turn(next_state, player, move)
            next_x = evaluate_heuristics(next_state, player)
            next_net = np.dot(weights, next_x)
            next_V = sigmoid(next_net)
            delta = alpha * (1 - V) * (gamma * next_V - V)
            weights += delta * np.array(x)

            # Update the state
            state = copy_board(next_state)
            player = 1 - player

    return weights


def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    global weights
    if weights is None:
        weights = torch.load('weights.pt')

    # Convert the board to a feature vector
    x = evaluate_heuristics(board, player)

    # Calculate the predicted value of each action
    values = []
    for action in legal_actions:
        next_board = copy_board(board)
        play_turn(next_board, player, action)
        next_x = evaluate_heuristics(next_board, player)
        net = np.dot(weights, next_x)
        V = sigmoid(net)
        values.append(V)

    # Select the action with the highest predicted value
    best_action = legal_actions[np.argmax(values)]

    # Update the board with the selected action
    play_turn(board, player, best_action)

    return best_action

# Example usage:
B = initial_board()
player = 0
alpha = 0.1
gamma = 0.9
num_episodes = 1000
weights = TD_0(B, player, alpha, gamma, num_episodes)
torch.save(weights, 'weights.pt')
