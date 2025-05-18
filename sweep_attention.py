# sweep_attention.py
import wandb
from train_attention import main

sweep_config = {
    "method": "bayes",
    "metric": {"name": "dev_accuracy", "goal": "maximize"},
    "parameters": {
        "embed_dim":      {"values": [64,128,256]},
        "hidden_dim":     {"values": [64,128,256]},
        "cell_type":      {"values": ["GRU","LSTM"]},
        "dropout":        {"values": [0.1,0.2,0.3]},
        "beam_size":      {"values": [3,5]},
        "epochs":         {"value": 10},
        "batch_size":     {"value": 64},
        "lr":             {"value": 1e-3},
        "max_len":        {"value": 32},
        "encoder_layers": {"value": 1},
        "decoder_layers": {"value": 1}
    }
}

sweep_id = wandb.sweep(sweep_config, project="assignment-03-attention")
wandb.agent(sweep_id, function=main, count=20)
