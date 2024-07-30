import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple
import wandb
from tqdm import tqdm


class PPOTrainer:
    def __init__(self, env, alice, bob, eve, config: Dict):
        self.env = env
        self.alice = alice
        self.bob = bob
        self.eve = eve
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alice.to(self.device)
        self.bob.to(self.device)
        self.action_dim = len(env.companies)


        self.alice_optimizer = optim.Adam(self.alice.parameters(), lr=float(config['learning_rate']))
        self.bob_optimizer = optim.Adam(self.bob.parameters(), lr=float(config['learning_rate']))

        self.clip_param = float(config['clip_param'])
        self.ppo_epochs = int(config['ppo_epochs'])
        self.num_mini_batch = int(config['num_mini_batch'])
        self.value_loss_coef = float(config['value_loss_coef'])
        self.entropy_coef = float(config['entropy_coef'])
        self.max_grad_norm = float(config['max_grad_norm'])

    def _state_to_tensor(self, state: Dict) -> torch.Tensor:
        market_data = state['market_data']
        state_list = []
        for company in self.env.companies:
            company_data = [d for d in market_data if d['Ticker'] == company]
            if company_data:
                state_list.extend([
                    company_data[0]['Open'],
                    company_data[0]['Close'],
                    company_data[0]['Volume'],
                    state['alice_portfolio'][company],
                    state['bob_portfolio'][company]
                ])
            else:
                state_list.extend([0, 0, 0, 0, 0])
        state_list.append(state['cash'])
        return torch.FloatTensor(state_list).unsqueeze(0).to(self.device)

    def train(self, num_episodes: int):
        for episode in tqdm(range(num_episodes), desc="Episodes"):
            state = self.env.reset()
            done = False
            episode_reward = 0
            alice_messages, bob_messages = [], []
            alice_thoughts, bob_thoughts = [], []
            eve_interjections = []

            pbar = tqdm(total=self.env.max_days, desc=f"Episode {episode+1} Steps", leave=False)
            while not done:
                    state_tensor = self._state_to_tensor(state)
                    
                    # Alice's turn
                    alice_action, alice_log_prob, alice_value = self.alice.act(state_tensor)
                    alice_message = self.alice.generate_message(state, state['alice_insider_info'])
                    alice_thought = self.alice.generate_thought(state, state['alice_insider_info'])
                    
                    # Eve's analysis of Alice's communication
                    eve_interjection_alice = self.eve.interject(alice_message, alice_thought)
                    
                    # Bob's turn
                    bob_action, bob_log_prob, bob_value = self.bob.act(state_tensor)
                    bob_interpretation = self.bob.interpret_message(alice_message)
                    bob_message = self.bob.generate_message(state, state['bob_insider_info'])
                    bob_thought = self.bob.generate_thought(state, state['bob_insider_info'])
                    
                    # Eve's analysis of Bob's communication
                    eve_interjection_bob = self.eve.interject(bob_message, bob_thought)
                    
                    actions = {
                        'alice': self._action_to_dict(alice_action),
                        'bob': self._action_to_dict(bob_action)
                    }

                    next_state, reward, done, _ = self.env.step(actions)
                    episode_reward += reward

                    # Store transition for PPO update
                    self._store_transition(state, actions, reward, next_state, done, 
                                        alice_log_prob, bob_log_prob, alice_value, bob_value)

                    state = next_state

                    # Store communications
                    alice_messages.append(alice_message)
                    bob_messages.append(bob_message)
                    alice_thoughts.append(alice_thought)
                    bob_thoughts.append(bob_thought)
                    eve_interjections.append(eve_interjection_alice)
                    eve_interjections.append(eve_interjection_bob)


                    # Perform PPO update
                    alice_loss, bob_loss = self._update_ppo()

                    # Log to wandb
                    wandb.log({
                        "episode": episode,
                        "reward": episode_reward,
                        "alice_loss": alice_loss,
                        "bob_loss": bob_loss,
                        "alice_messages": alice_messages,
                        "bob_messages": bob_messages,
                        "alice_thoughts": alice_thoughts,
                        "bob_thoughts": bob_thoughts,
                        "eve_interjections": eve_interjections
                    })

                    print(f"Episode {episode + 1}, Reward: {episode_reward}")
                    pbar.update(1)
            pbar.close()

    def _action_to_dict(self, action: int) -> Dict[str, Dict[str, int]]:
        return {
            'alice': {'action': action},
            'bob': {'action': action}
        }

    def _store_transition(self, state, actions, reward, next_state, done, 
                          alice_log_prob, bob_log_prob, alice_value, bob_value):
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.alice_log_probs.append(alice_log_prob)
        self.bob_log_probs.append(bob_log_prob)
        self.alice_values.append(alice_value)
        self.bob_values.append(bob_value)

    def _update_ppo(self):
        # Calculate advantages and returns
        alice_advantages, alice_returns = self._compute_advantages_and_returns(self.alice_values)
        bob_advantages, bob_returns = self._compute_advantages_and_returns(self.bob_values)

        # Perform PPO update for Alice
        alice_loss = self._ppo_update(self.alice, self.alice_optimizer, 
                                      self.states, self.actions, alice_advantages, 
                                      alice_returns, self.alice_log_probs)

        # Perform PPO update for Bob
        bob_loss = self._ppo_update(self.bob, self.bob_optimizer, 
                                    self.states, self.actions, bob_advantages, 
                                    bob_returns, self.bob_log_probs)

        # Clear stored transitions
        self._clear_storage()

        return alice_loss, bob_loss

    def _compute_advantages_and_returns(self, values):
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = self.rewards[i] + self.config['gamma'] * next_value * (1 - self.dones[i]) - values[i]
            gae = delta + self.config['gamma'] * self.config['lambda'] * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        return advantages, returns

    def _ppo_update(self, agent, optimizer, states, actions, advantages, returns, old_log_probs):
        for _ in range(self.ppo_epochs):
            for batch in self._get_minibatches(states, actions, advantages, returns, old_log_probs):
                state_batch, action_batch, advantage_batch, return_batch, old_log_prob_batch = batch
                
                new_log_probs, state_values = agent(state_batch)
                new_log_probs = new_log_probs.gather(1, action_batch)
                
                ratio = (new_log_probs - old_log_prob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_batch
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, return_batch)
                
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * new_log_probs.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                optimizer.step()
        
        return loss.item()

    def _get_minibatches(self, states, actions, advantages, returns, old_log_probs):
        batch_size = len(states)
        minibatch_size = batch_size // self.num_mini_batch
        indices = np.random.permutation(batch_size)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.tensor([states[i] for i in batch_indices], dtype=torch.float32).to(self.device),
                torch.tensor([actions[i] for i in batch_indices], dtype=torch.long).to(self.device),
                advantages[batch_indices],
                returns[batch_indices],
                old_log_probs[batch_indices]
            )

    def _clear_storage(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.alice_log_probs = []
        self.bob_log_probs = []
        self.alice_values = []
        self.bob_values = []