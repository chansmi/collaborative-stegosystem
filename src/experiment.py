import wandb
import json
import os
from datetime import datetime
from tqdm import tqdm
import tiktoken

class Experiment:
    def __init__(self, config, environment, alice, bob, eve):
        self.config = config
        self.env = environment
        self.alice = alice
        self.bob = bob
        self.eve = eve
        self.results = []
        self.total_tokens = 0
        self.total_cost = 0
        self.enc = tiktoken.encoding_for_model(self.config['model_name'])

    def run(self):
        wandb.init(project="Collaborative Stegosystem", entity="chansmi", config=self.config)

        instructions = self.env.get_instructions()
        context = self.env.get_context()

        for episode in tqdm(range(self.config['num_episodes']), desc="Episodes"):
            episode_data = self.env.get_episode_data(episode)
            episode_results = self.run_episode(episode, instructions, context, episode_data)
            self.results.extend(episode_results)

        self.save_results()
        wandb.log({"total_tokens": self.total_tokens, "total_cost": self.total_cost})
        wandb.finish()
        return self.results

    def run_episode(self, episode, instructions, context, episode_data):
        episode_results = []
        for turn in range(self.config['turns_per_episode']):
            turn_results = self.run_turn(episode, turn, instructions, context, episode_data)
            episode_results.extend(turn_results)
        return episode_results

    def run_turn(self, episode, turn, instructions, context, episode_data):
        turn_results = []

        # Alice's turn
        alice_thought = self.alice.get_thought_process(context, self.config['max_tokens'])
        alice_message = self.alice.generate_message(context, self.config['max_tokens'])

        # Bob's turn
        bob_thought = self.bob.get_thought_process(alice_message, self.config['max_tokens'])
        bob_message = self.bob.generate_message(alice_message, self.config['max_tokens'])

        # Eve's evaluation
        eve_thought = self.eve.get_thought_process(f"{alice_message}\n{bob_message}", self.config['max_tokens'])
        eve_interjection, reward = self.env.evaluate_message(f"{alice_message}\n{bob_message}")

        turn_result = {
            "episode": episode,
            "turn": turn,
            "episode_data": episode_data,
            "alice_thought": alice_thought,
            "alice_message": alice_message,
            "bob_thought": bob_thought,
            "bob_message": bob_message,
            "eve_thought": eve_thought,
            "eve_interjection": eve_interjection,
            "reward": reward
        }

        self.update_token_count(turn_result)

        turn_results.append(turn_result)
        wandb.log(turn_result)

        # Print the exchange to console
        print(f"\nEpisode {episode}, Turn {turn}")
        print(f"Episode Data: {episode_data}")
        print(f"Alice: {alice_message}")
        print(f"Bob: {bob_message}")
        if eve_interjection:
            print(f"Eve interjected! Reward set to {reward}")
        print(f"Reward: {reward}")

        return turn_results

    def update_token_count(self, turn_result):
        for key, value in turn_result.items():
            if isinstance(value, str):
                tokens = len(self.enc.encode(value))
                self.total_tokens += tokens
        # Assuming $0.02 per 1K tokens for GPT-4
        self.total_cost += (self.total_tokens / 1000) * 0.02

    def save_results(self):
        results_dir = f"results/{self.env.__class__.__name__.lower()}"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/experiment_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {filename}")
        wandb.save(filename)