import wandb
from train import main

sweep_config = {
    "method": "bayes",
    "metric": {"name": "dev_accuracy", "goal": "maximize"},
    "parameters": {
        "embed_dim":        {"values": [16, 32, 64, 256]},
        "hidden_dim":       {"values": [16, 32, 64, 256]},
        "encoder_layers":   {"values": [1, 2, 3]},
        "decoder_layers":   {"values": [1, 2, 3]},
        "cell_type":        {"values": ["RNN","GRU","LSTM"]},
        "dropout":          {"values": [0.2, 0.3]},
        "beam_size":        {"values": [1, 3, 5]},
        "max_len":          {"value": 32},
        "batch_size":       {"value": 64},
        "lr":               {"value": 1e-3},
        "epochs":           {"value": 10}
    }
}

sweep_id = wandb.sweep(sweep_config, project="assignment-03")
wandb.agent(sweep_id, function=main, count=20)
