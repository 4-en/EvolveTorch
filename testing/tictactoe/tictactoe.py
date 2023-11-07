# text based tic tac toe game
import random


class TicTacToe:
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 is empty, 1 is X, 2 is O
        self.turn = 1  # 1 is X, 2 is O
        self.winner = 0  # 0 is no winner, 1 is X, 2 is O
        self.game_over = False

    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 is empty, 1 is X, 2 is O
        self.turn = 1  # 1 is X, 2 is O
        self.winner = 0  # 0 is no winner, 1 is X, 2 is O
        self.game_over = False

    def get_board(self):
        return self.board

    def get_turn(self):
        return self.turn

    def get_winner(self):
        return self.winner

    def is_game_over(self):
        return self.game_over

    def get_valid_moves(self):
        valid_moves = []
        for i in range(9):
            if self.board[i] == 0:
                valid_moves.append(i)
        return valid_moves

    def make_move(self, move):
        if self.board[move] == 0:
            self.board[move] = self.turn
            self.turn = 3 - self.turn
            self.check_winner()

    def check_winner(self):
        # check rows
        for i in range(3):
            if self.board[i * 3] == self.board[i * 3 + 1] and self.board[i * 3 + 1] == self.board[i * 3 + 2] and self.board[i * 3] != 0:
                self.winner = self.board[i * 3]
                self.game_over = True
        # check columns
        for i in range(3):
            if self.board[i] == self.board[i + 3] and self.board[i + 3] == self.board[i + 6] and self.board[i] != 0:
                self.winner = self.board[i]
                self.game_over = True
        # check diagonals
        if self.board[0] == self.board[4] and self.board[4] == self.board[8] and self.board[0] != 0:
            self.winner = self.board[0]
            self.game_over = True
        if self.board[2] == self.board[4] and self.board[4] == self.board[6] and self.board[2] != 0:
            self.winner = self.board[2]
            self.game_over = True

    def print_board(self):
        for i in range(3):
            print(self.board[i * 3], self.board[i * 3 + 1], self.board[i * 3 + 2])
        print()


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def get_move(self):
        return random.choice(self.game.get_valid_moves())
    
    def get_name(self):
        return "Random Player"
    
    def reset(self):

        return
    
    def get_game(self):
        return self.game
    
    def set_game(self, game):
        self.game = game


class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def get_move(self):
        while True:
            move = input("Enter a move: ")
            if move.isdigit():
                move = int(move)
                if move in self.game.get_valid_moves():
                    return move
            print("Invalid move")

    def get_name(self):
        return "Human Player"
    
    def reset(self):
        return
    
    def get_game(self):
        return self.game
    
    def set_game(self, game):
        self.game = game


def main():
    game = TicTacToe()
    player1 = HumanPlayer(game)
    player2 = RandomPlayer(game)
    while True:
        game.reset()
        while not game.is_game_over():
            game.print_board()
            if game.get_turn() == 1:
                move = player1.get_move()
            else:
                move = player2.get_move()
            game.make_move(move)
        game.print_board()
        if game.get_winner() == 0:
            print("Tie game")
        else:
            print("Player", game.get_winner(), "wins!")
        input("Press enter to play again")


if __name__ == "__main__":
    main()