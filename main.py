from game import TicTacToe
from train import train_agent


def play_game(agent):

    game = TicTacToe()

    # Disable exploration during the real game
    agent.exploration_rate = 0

    print("\n======================")
    print("     TIC-TAC-TOE")
    print("======================")

    print("You are X")
    print("AI is O")

    print("\nBoard positions:")

    print(" 1 | 2 | 3 ")
    print("---+---+---")
    print(" 4 | 5 | 6 ")
    print("---+---+---")
    print(" 7 | 8 | 9 ")

    while True:

        game.display()

        # =========================================
        # HUMAN MOVE
        # =========================================

        available = game.available_actions()

        while True:

            try:

                position = int(
                    input("Enter position (1-9): ")
                ) - 1

                if position in available:
                    break

                print(
                    "Invalid position. "
                    "Choose an empty position."
                )

            except ValueError:

                print(
                    "Please enter a number "
                    "from 1 to 9."
                )

        game.make_move(
            position,
            "X"
        )

        result = game.check_winner()

        if result == "X":

            game.display()

            print(
                "Congratulations! "
                "You win!"
            )

            break

        if result == "Draw":

            game.display()

            print("It's a draw!")

            break

        # =========================================
        # AI MOVE
        # =========================================

        state = game.get_state()

        available = game.available_actions()

        action = agent.choose_action(
            state,
            available
        )

        game.make_move(
            action,
            "O"
        )

        print(
            "AI selected position:",
            action + 1
        )

        result = game.check_winner()

        if result == "O":

            game.display()

            print("AI wins!")

            break

        if result == "Draw":

            game.display()

            print("It's a draw!")

            break


if __name__ == "__main__":

    print("\n======================")
    print("   TRAINING AI")
    print("======================")

    print(
        "\nTraining for 200,000 games..."
    )

    print(
        "Please wait...\n"
    )

    agent = train_agent(
        200000
    )

    print(
        "\nAI is ready!"
    )

    play_game(agent)
