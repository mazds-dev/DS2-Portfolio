# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# GRIDWORLD TAB — Q-Learning Agent with Interactive Training
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# ENVIRONMENT AND AGENT CLASSES
# ============================================================
class GridWorld:
    def __init__(self):
        self.height = 6
        self.width = 6
        self.grid = np.full((self.height, self.width), -1.0)
        self.walls = [(1, 2), (2, 2), (3, 2), (2, 4), (3, 4), (4, 4)]
        self.start_location = (5, 0)
        self.current_location = self.start_location
        self.goal_location = (0, 5)
        self.grid[self.goal_location[0], self.goal_location[1]] = 10.0
        self.terminal_states = [self.goal_location]
        self.actions = ["NORTH", "SOUTH", "EAST", "WEST"]

    def get_available_actions(self):
        return self.actions

    def get_reward(self, new_location):
        return self.grid[new_location[0], new_location[1]]

    def make_step(self, action):
        last_location = self.current_location
        if action == "NORTH":
            new_location = (last_location[0] - 1, last_location[1])
        elif action == "SOUTH":
            new_location = (last_location[0] + 1, last_location[1])
        elif action == "EAST":
            new_location = (last_location[0], last_location[1] + 1)
        elif action == "WEST":
            new_location = (last_location[0], last_location[1] - 1)
        else:
            new_location = last_location

        if (0 <= new_location[0] < self.height and
                0 <= new_location[1] < self.width and
                new_location not in self.walls):
            self.current_location = new_location
            return self.get_reward(new_location)
        return -1.0

    def check_state(self):
        if self.current_location in self.terminal_states:
            return "TERMINAL"
        return "ONGOING"

    def reset(self):
        self.current_location = self.start_location
        return self.current_location


class Q_Agent:
    def __init__(self, environment, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.environment = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        for x in range(environment.height):
            for y in range(environment.width):
                self.q_table[(x, y)] = {a: 0 for a in environment.actions}

    def choose_action(self, available_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(available_actions)
        q_values = self.q_table[self.environment.current_location]
        max_value = max(q_values.values())
        best_actions = [k for k, v in q_values.items() if v == max_value]
        return np.random.choice(best_actions)

    def learn(self, old_state, reward, new_state, action):
        max_q_new = max(self.q_table[new_state].values())
        old_q = self.q_table[old_state][action]
        self.q_table[old_state][action] = (
            (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_q_new)
        )


# ============================================================
# VISUALISATION FUNCTIONS (smaller figures)
# ============================================================
def plot_grid_with_agent(env, show_path=False, path=None):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.walls:
                colour = "#2c3e50"
            elif (i, j) == env.start_location:
                colour = "#2ecc71"
            elif (i, j) == env.goal_location:
                colour = "#f39c12"
            else:
                colour = "#ecf0f1"

            rect = patches.Rectangle(
                (j, env.height - 1 - i), 1, 1,
                linewidth=1.5, edgecolor="black", facecolor=colour,
            )
            ax.add_patch(rect)

            if (i, j) == env.start_location:
                ax.text(j + 0.5, env.height - 1 - i + 0.15, "START",
                        ha="center", fontsize=7, fontweight="bold")
            elif (i, j) == env.goal_location:
                ax.text(j + 0.5, env.height - 1 - i + 0.5, "GOAL",
                        ha="center", va="center", fontsize=8, fontweight="bold")

    ai, aj = env.current_location
    ax.plot(aj + 0.5, env.height - 1 - ai + 0.5, "o",
            markersize=20, color="#e74c3c", markeredgecolor="black")

    if show_path and path:
        for (pi, pj) in path:
            if (pi, pj) != env.start_location and (pi, pj) != env.goal_location:
                circle = patches.Circle(
                    (pj + 0.5, env.height - 1 - pi + 0.5),
                    0.12, facecolor="#3498db", alpha=0.4,
                )
                ax.add_patch(circle)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_policy(agent, env):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    arrow_map = {
        "NORTH": (0, 0.3),
        "SOUTH": (0, -0.3),
        "EAST": (0.3, 0),
        "WEST": (-0.3, 0),
    }

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.walls:
                colour = "#2c3e50"
            elif (i, j) == env.start_location:
                colour = "#2ecc71"
            elif (i, j) == env.goal_location:
                colour = "#f39c12"
            else:
                colour = "#ecf0f1"

            rect = patches.Rectangle(
                (j, env.height - 1 - i), 1, 1,
                linewidth=1.5, edgecolor="black", facecolor=colour,
            )
            ax.add_patch(rect)

            if (i, j) not in env.walls and (i, j) != env.goal_location:
                q_values = agent.q_table[(i, j)]
                if max(q_values.values()) != 0:
                    best_action = max(q_values, key=q_values.get)
                    dx, dy = arrow_map[best_action]
                    ax.arrow(
                        j + 0.5, env.height - 1 - i + 0.5, dx, dy,
                        head_width=0.13, head_length=0.13,
                        fc="blue", ec="blue", linewidth=1.5,
                    )

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Learned Policy", fontsize=10)
    return fig


def train_episodes(env, agent, n_episodes=500, max_steps=200):
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        cumulative = 0
        step = 0
        while step < max_steps:
            old_state = env.current_location
            action = agent.choose_action(env.actions)
            reward = env.make_step(action)
            new_state = env.current_location
            agent.learn(old_state, reward, new_state, action)
            cumulative += reward
            step += 1
            if env.check_state() == "TERMINAL":
                break
        rewards.append(cumulative)
    return rewards


def get_optimal_path(env, agent, max_steps=30):
    env.reset()
    path = [env.current_location]
    for _ in range(max_steps):
        q_values = agent.q_table[env.current_location]
        if max(q_values.values()) == 0:
            break
        best_action = max(q_values, key=q_values.get)
        env.make_step(best_action)
        path.append(env.current_location)
        if env.check_state() == "TERMINAL":
            break
    return path


# ============================================================
# STREAMLIT RENDER
# ============================================================
def render():
    st.header("🎮 Gridworld — Q-Learning Agent")
    st.markdown(
        "Watch a **Reinforcement Learning** agent learn to navigate a grid "
        "with obstacles. Adjust the hyperparameters below and click "
        "**Train** to see how they affect learning."
    )

    # --- Hyperparameter controls (2x2 grid) ---
    st.subheader("⚙️ Hyperparameters")

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider(
            "Learning rate (α)", 0.01, 1.0, 0.1, 0.01,
            help="How much new experience overrides old knowledge",
        )
        epsilon = st.slider(
            "Exploration rate (ε)", 0.0, 1.0, 0.1, 0.01,
            help="Probability of random action",
        )
    with col2:
        gamma = st.slider(
            "Discount factor (γ)", 0.0, 1.0, 0.9, 0.05,
            help="Importance of future rewards",
        )
        n_episodes = st.slider(
            "Episodes", 100, 1000, 500, 100,
            help="Number of training episodes",
        )

    train_button = st.button("🚀 Train Agent", type="primary", use_container_width=True)

    if "trained_agent" not in st.session_state:
        st.session_state.trained_agent = None
        st.session_state.trained_env = None
        st.session_state.rewards = None

    if train_button:
        env = GridWorld()
        agent = Q_Agent(env, epsilon=epsilon, alpha=alpha, gamma=gamma)

        progress = st.progress(0)
        status = st.empty()

        rewards = []
        chunk_size = max(10, n_episodes // 20)
        for chunk_start in range(0, n_episodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_episodes)
            chunk_rewards = train_episodes(env, agent, n_episodes=chunk_end - chunk_start)
            rewards.extend(chunk_rewards)
            progress.progress(chunk_end / n_episodes)
            status.text(
                f"Episode {chunk_end}/{n_episodes} — "
                f"recent avg reward: {np.mean(rewards[-20:]):.2f}"
            )

        status.empty()
        progress.empty()

        st.session_state.trained_agent = agent
        st.session_state.trained_env = env
        st.session_state.rewards = rewards

        st.success(
            f"✅ Training complete! Final 50 episodes average: "
            f"**{np.mean(rewards[-50:]):.2f}**"
        )

    # --- Display results ---
    if st.session_state.trained_agent is not None:
        st.markdown("---")

        agent = st.session_state.trained_agent
        env = st.session_state.trained_env
        rewards = st.session_state.rewards

        # Rewards chart
        st.subheader("📈 Learning progress")
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.plot(rewards, alpha=0.3, color="#3498db", label="Raw")
        if len(rewards) >= 20:
            smoothed = np.convolve(rewards, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, len(rewards)), smoothed,
                    color="#2ecc71", linewidth=2, label="Moving average (20)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        # Policy
        st.subheader("🧭 Learned policy")
        fig = plot_policy(agent, env)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        # Optimal path
        st.subheader("🎯 Optimal path")
        path = get_optimal_path(env, agent)

        if env.check_state() == "TERMINAL":
            st.success(
                f"The agent found a path to the goal in **{len(path) - 1} steps**!"
            )
        else:
            st.warning(
                "The agent could not reach the goal. Try training for more episodes."
            )

        env.reset()
        fig = plot_grid_with_agent(env, show_path=True, path=path)
        st.pyplot(fig, use_container_width=False)
        plt.close()
    else:
        st.markdown("---")
        st.subheader("🏁 The environment")
        env = GridWorld()
        fig = plot_grid_with_agent(env)
        st.pyplot(fig, use_container_width=False)
        plt.close()
        st.info(
            "👆 Adjust the hyperparameters above and click **Train Agent** "
            "to start learning."
        )

    with st.expander("ℹ️ How Q-Learning works"):
        st.markdown(
            """
            **Q-Learning** is a Reinforcement Learning algorithm where an
            agent learns through **trial and error**. Unlike supervised
            learning, it is never told the correct action — it only receives
            rewards (positive or negative) after each action.

            **The core idea:**
            1. The agent starts with no knowledge — it moves randomly
            2. Every time it reaches the goal, it remembers which actions
               led there
            3. Gradually it learns a **Q-table** — a map of "how good" each
               action is from each position
            4. After enough episodes, the Q-table shows the optimal path

            **Hyperparameter effects:**
            - **α** — too high makes learning unstable; too low is slow
            - **γ** — low values focus on immediate rewards; high values
              encourage long-term planning
            - **ε** — balances exploration and exploitation

            **Try experimenting:**
            - What happens with ε = 0? (No exploration)
            - What happens with ε = 1? (Fully random — no learning)
            - What about γ = 0? (Only immediate reward matters)
            """
        )

    with st.expander("💻 Source Code — Q-Learning Update"):
        st.code("""class Q_Agent():
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1.0):
        self.q_table = dict()
        # Initialise Q-table with zeros for all states and actions
        for x in range(environment.height):
            for y in range(environment.width):
                self.q_table[(x, y)] = {a: 0 for a in environment.actions}

    def choose_action(self, available_actions):
        # Epsilon-greedy: explore with probability epsilon
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            q_values = self.q_table[self.environment.current_location]
            max_value = max(q_values.values())
            best_actions = [k for k, v in q_values.items() if v == max_value]
            action = np.random.choice(best_actions)
        return action

    def learn(self, old_state, reward, new_state, action):
        # Bellman equation update
        max_q_new = max(self.q_table[new_state].values())
        current_q = self.q_table[old_state][action]
        self.q_table[old_state][action] = (
            (1 - self.alpha) * current_q +
            self.alpha * (reward + self.gamma * max_q_new)
        )""", language="python")