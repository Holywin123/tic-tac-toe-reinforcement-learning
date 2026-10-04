import random


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995
    ):

        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9

        return self.q_table[state]

    def choose_action(self, state, available_actions):

        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        q_values = self.get_q_values(state)

        best_action = max(
            available_actions,
            key=lambda action: q_values[action]
        )

        return best_action

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        available_actions
    ):

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if available_actions:

            next_q = max(
                self.get_q_values(next_state)[a]
                for a in available_actions
            )

        else:
            next_q = 0

        new_q = current_q + self.learning_rate * (
            reward
            + self.discount_factor * next_q
            - current_q
        )

        q_values[action] = new_q

    def decay_exploration(self):

        self.exploration_rate *= self.exploration_decay