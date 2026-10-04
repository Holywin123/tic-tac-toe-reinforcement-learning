from game import TicTacToe
from q_learning import QLearningAgent
import random


def train_agent(episodes=50000):

    agent = QLearningAgent()

    for episode in range(episodes):

        game = TicTacToe()

        state = game.reset()

        while True:

            # AI chooses a move
            available = game.available_actions()

            action = agent.choose_action(
                state,
                available
            )

            game.make_move(action, "O")

            result = game.check_winner()

            # AI wins
            if result == "O":

                reward = 10

                next_state = game.get_state()

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    []
                )

                break

            # Draw
            if result == "Draw":

                reward = 1

                next_state = game.get_state()

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    []
                )

                break

            # Random opponent
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

                reward = -10

                next_state = game.get_state()

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    []
                )

                break

            # Draw
            if result == "Draw":

                reward = 1

                next_state = game.get_state()

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    []
                )

                break

            # Continue learning
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

    print("Training completed!")

    print(
        "Number of learned states:",
        len(agent.q_table)
    )

    return agent


if __name__ == "__main__":

    train_agent(50000)