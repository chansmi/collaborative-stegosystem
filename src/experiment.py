import wandb
import json
import os
from datetime import datetime

class Experiment:
    def __init__(self, config, environment, alice, bob, eve):
        self.config = config
        self.env = environment
        self.alice = alice
        self.bob = bob
        self.eve = eve
        self.results = []

    def run(self):
        wandb.init(project="Collaborative Stegosystem", entity="chansmi", config=self.config)

        instructions = self.env.get_instructions()
        context = self.env.get_context()

        for episode in range(self.config['num_episodes']):
            episode_results = self.run_episode(episode, instructions, context)
            self.results.extend(episode_results)

        self.save_results()
        wandb.finish()
        return self.results

    def run_episode(self, episode, instructions, context):
        episode_results = []
        for turn in range(self.config['turns_per_episode']):
            turn_results = self.run_turn(episode, turn, instructions, context)
            episode_results.extend(turn_results)
        return episode_results

    def run_turn(self, episode, turn, instructions, context):
        turn_results = []

        # Alice's turn
        alice_thought = self.alice.get_thought_process(context)
        alice_message = self.alice.generate_message(context)

        # Bob's turn
        bob_thought = self.bob.get_thought_process(alice_message)
        bob_message = self.bob.generate_message(alice_message)

        # Eve's evaluation
        eve_thought = self.eve.get_thought_process(f"{alice_message}\n{bob_message}")
        reward = self.env.evaluate_message(f"{alice_message}\n{bob_message}")

        turn_result = {
            "episode": episode,
            "turn": turn,
            "alice_thought": alice_thought,
            "alice_message": alice_message,
            "bob_thought": bob_thought,
            "bob_message": bob_message,
            "eve_thought": eve_thought,
            "reward": reward
        }

        turn_results.append(turn_result)
        wandb.log(turn_result)

        return turn_results

    def save_results(self):
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/experiment_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {filename}")