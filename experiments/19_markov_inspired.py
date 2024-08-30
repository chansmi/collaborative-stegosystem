import torch as th
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Custom Environment for Simple Communication Task
class SimpleCommunicationEnv(gym.Env):
    def __init__(self):
        super(SimpleCommunicationEnv, self).__init__()
        # Sender's observation space (payload is 0 or 1)
        self.observation_space = spaces.Discrete(2)
        # Sender's action space (choosing one of four colors)
        self.action_space = spaces.Discrete(4)
        # Receiver's action space (guessing 0 or 1)
        self.receiver_action_space = spaces.Discrete(2)
        # Initialize state and sender's action
        self.state = None
        self.sender_action = None

    def reset(self):
        # Randomly generate payload (0 or 1)
        self.state = self.observation_space.sample()
        self.sender_action = None
        return np.array([self.state])

    def step(self, action):
        if self.sender_action is None:
            # Sender's turn
            self.sender_action = action
            # Receiver will observe the color corresponding to sender's action
            return np.array([self.sender_action]), 0, False, {}
        else:
            # Receiver's turn
            receiver_action = action
            reward = 1 if receiver_action == self.state else -1
            done = True
            return np.array([self.state, self.sender_action]), reward, done, {}

    def render(self, mode="human"):
        print(f"Payload: {self.state}, Sender chose color {self.sender_action}")

    def close(self):
        pass

# Define the custom feature extractor
class SimpleFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(SimpleFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = th.nn.Sequential(
            th.nn.Linear(observation_space.shape[0], features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

# Define the policy using the custom feature extractor
class SimpleActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(SimpleActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=SimpleFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )

# Create and wrap the environment
env = make_vec_env(lambda: SimpleCommunicationEnv(), n_envs=1)

# Initialize the PPO model with the custom policy
model = PPO(SimpleActorCriticPolicy, env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()