from game import TicTacToe
from q_learning import QLearningAgent
import random
import pickle


def train_agent(episodes=200000):

    # Agent 1 learns to play X
    agent_x = QLearningAgent()

    # Agent 2 learns to play O
    agent_o = QLearningAgent()

    for episode in range(episodes):

        game = TicTacToe()

        state = game.reset()

        # Randomly decide who starts
        current_player = random.choice(
            ["X", "O"]
        )

        while True:

            # =========================================
            # X PLAYER
            # =========================================

            if current_player == "X":

                available = game.available_actions()

                action = agent_x.choose_action(
                    state,
                    available
                )

                game.make_move(
                    action,
                    "X"
                )

                result = game.check_winner()

                # X wins
                if result == "X":

                    agent_x.update(
                        state,
                        action,
                        10,
                        game.get_state(),
                        []
                    )

                    break

                # Draw
                if result == "Draw":

                    agent_x.update(
                        state,
                        action,
                        5,
                        game.get_state(),
                        []
                    )

                    break

                # Change to O
                next_state = game.get_state()

                next_actions = game.available_actions()

                agent_x.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_actions
                )

                state = next_state

                current_player = "O"

            # =========================================
            # O PLAYER
            # =========================================

            else:

                available = game.available_actions()

                action = agent_o.choose_action(
                    state,
                    available
                )

                game.make_move(
                    action,
                    "O"
                )

                result = game.check_winner()

                # O wins
                if result == "O":

                    agent_o.update(
                        state,
                        action,
                        10,
                        game.get_state(),
                        []
                    )

                    break

                # Draw
                if result == "Draw":

                    agent_o.update(
                        state,
                        action,
                        5,
                        game.get_state(),
                        []
                    )

                    break

                # Change to X
                next_state = game.get_state()

                next_actions = game.available_actions()

                agent_o.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_actions
                )

                state = next_state

                current_player = "X"

        # =============================================
        # DECAY EXPLORATION
        # =============================================

        agent_x.decay_exploration()
        agent_o.decay_exploration()

        # =============================================
        # TRAINING PROGRESS
        # =============================================

        if (episode + 1) % 10000 == 0:

            print(
                f"Training episode: "
                f"{episode + 1}/{episodes}"
            )

    # =============================================
    # SAVE THE O AGENT
    # =============================================

    with open(
        "q_table.pkl",
        "wb"
    ) as file:

        pickle.dump(
            agent_o.q_table,
            file
        )

    print("\nTraining completed!")

    print(
        "O Agent learned states:",
        len(agent_o.q_table)
    )

    print(
        "Final exploration rate:",
        agent_o.exploration_rate
    )

    return agent_o


if __name__ == "__main__":

    train_agent(200000)
