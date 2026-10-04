class TicTacToe:

    def __init__(self):
        self.board = [" "] * 9

    def reset(self):
        self.board = [" "] * 9
        return self.get_state()

    def get_state(self):
        return "".join(self.board)

    def available_actions(self):
        return [
            i for i, cell in enumerate(self.board)
            if cell == " "
        ]

    def make_move(self, position, player):

        if self.board[position] == " ":
            self.board[position] = player
            return True

        return False

    def check_winner(self):

        winning_positions = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6)
        ]

        for a, b, c in winning_positions:

            if self.board[a] != " ":

                if self.board[a] == self.board[b] == self.board[c]:
                    return self.board[a]

        if " " not in self.board:
            return "Draw"

        return None

    def display(self):

        print()

        print(
            f" {self.board[0]} | "
            f"{self.board[1]} | "
            f"{self.board[2]} "
        )

        print("---+---+---")

        print(
            f" {self.board[3]} | "
            f"{self.board[4]} | "
            f"{self.board[5]} "
        )

        print("---+---+---")

        print(
            f" {self.board[6]} | "
            f"{self.board[7]} | "
            f"{self.board[8]} "
        )

        print()