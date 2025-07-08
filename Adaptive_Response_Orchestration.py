import numpy as np

def Reinforcement_Learning():
    states = ['Low Threat', 'Moderate Threat', 'High Threat']  # S
    actions = ['No Action', 'Isolate VM', 'Increase Firewall Rules', 'Reallocate Resources']  # A

    # Transition probabilities P(s'|s,a) simplified to depend only on s here
    transition_probabilities = {
        'Low Threat': [0.9, 0.08, 0.02],
        'Moderate Threat': [0.2, 0.6, 0.2],
        'High Threat': [0.05, 0.15, 0.8]
    }

    # Reward Function R(s,a)
    reward_table = {
        'No Action': 1,
        'Isolate VM': 6,
        'Increase Firewall Rules': 4,
        'Reallocate Resources': 3
    }

    gamma = 0.9  # Discount factor γ

    # --- Define Agent with Q-Table (Q(s,a)) ---
    class DQNAgent:
        def __init__(self, states, actions, lr=0.1):
            self.q_table = {state: np.zeros(len(actions)) for state in states}
            self.states = states
            self.actions = actions
            self.lr = lr

        def policy(self, state):
            """π(s) = argmax_a Q(s,a)"""
            return np.argmax(self.q_table[state])

        def update(self, state, action_idx, reward, next_state):
            """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]"""
            current_q = self.q_table[state][action_idx]
            max_future_q = np.max(self.q_table[next_state])
            self.q_table[state][action_idx] += self.lr * (reward + gamma * max_future_q - current_q)

        def get_action_name(self, action_idx):
            return self.actions[action_idx]

    # --- Initialize Agent ---
    agent = DQNAgent(states, actions)

    # --- Train for 100 Episodes ---
    for episode in range(100):
        state = np.random.choice(states)
        action_idx = agent.policy(state)
        action = agent.get_action_name(action_idx)
        reward = reward_table[action]
        next_state = np.random.choice(states, p=transition_probabilities[state])
        agent.update(state, action_idx, reward, next_state)

        print(f"[EP {episode + 1:03d}] State: {state} -> Action: {action} -> Reward: {reward} -> Next: {next_state}")

    print("\n[INFO] RL-based Adaptive Response Learning Completed.")

    # --- Optional: Print Final Q-Table ---
    print("\nFinal Q-Table (Q(s, a)):")
    for state in states:
        print(f"{state}: {dict(zip(actions, agent.q_table[state]))}")
