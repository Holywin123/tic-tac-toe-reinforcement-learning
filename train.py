
from game import TicTacToe
from q_learning import QLearningAgent
import random
import pickle


def train_agent(episodes=100000):

    agent = QLearningAgent()

    for episode in range(episodes):

        game = TicTacToe()

        state = game.reset()

        while True:

            # -------------------------
            # AI MOVE
            # -------------------------

            available = game.available_actions()

            action = agent.choose_action(
                state,
                available
            )

            game.make_move(
                action,
                "O"
            )

            result = game.check_winner()

            # AI wins
            if result == "O":

                agent.update(
                    state,
                    action,
                    10,
                    game.get_state(),
                    []
                )

                break

            # Draw
            if result == "Draw":

                agent.update(
                    state,
                    action,
                    5,
                    game.get_state(),
                    []
                )

                break

            # -------------------------
            # RANDOM OPPONENT MOVE
            # -------------------------

            opponent_actions = game.available_actions()

            opponent_action = random.choice(
                opponent_actions
            )

            game.make_move(
                opponent_action,
                "X"
            )

            result = game.check_winner()

            # Opponent wins
            if result == "X":

                agent.update(
                    state,
                    action,
                    -10,
                    game.get_state(),
                    []
                )

                break

            # Draw
            if result == "Draw":

                agent.update(
                    state,
                    action,
                    5,
                    game.get_state(),
                    []
                )

                break

            # -------------------------
            # CONTINUE LEARNING
            # -------------------------

            next_state = game.get_state()

            next_available = game.available_actions()

            agent.update(
                state,
                action,
                0,
                next_state,
                next_available
            )

            state = next_state

        agent.decay_exploration()

        # Show progress
        if (episode + 1) % 10000 == 0:

            print(
                f"Training episode: "
                f"{episode + 1}/{episodes}"
            )

    # Save trained AI
    with open(
        "q_table.pkl",
        "wb"
    ) as file:

        pickle.dump(
            agent.q_table,
            file
        )

    print("\nTraining completed!")

    print(
        "Number of learned states:",
        len(agent.q_table)
    )

    return agent


if __name__ == "__main__":

    train_agent(100000)

