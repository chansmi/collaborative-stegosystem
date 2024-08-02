# Collaborative Stegosystem

This project implements an experiment in collaborative steganography using reinforcement learning with language models. Two agents, Alice and Bob, attempt to share secret information while maximizing a reward function. An overseer, Eve, evaluates their communication and penalizes obvious information sharing.

Run main.py, make sure you have a hf and are logged in. Also make sure you are logged into wandb. 




Regarding train_sft.py and train_reward.py
You're correct that we moved away from these files in the recent updates. This was done to simplify the implementation and focus on the core PPO training loop. However, if you want to include Supervised Fine-Tuning (SFT) and a separate reward model, you can certainly reintegrate these components.

The decision to include or exclude SFT and a separate reward model depends on your specific requirements:

If you want to pre-train the models on a specific dataset before PPO training, you would use SFT.
If you want a more complex reward function that's learned from data rather than hand-crafted, you would use a separate reward model.

In your current setup, we're using the portfolio performance as the reward, which doesn't require a separate reward model. And we're starting with the base language models without additional fine-tuning.
If you decide you need these components, you can reintegrate them by:

Calling the SFT training function before starting the PPO training loop.
Training a reward model and using it in the compute_reward method instead of relying solely on the environment's reward.

For now, if the current setup meets your requirements, we can continue without these additional components. But if you feel they would be beneficial, we can certainly add them back in.

