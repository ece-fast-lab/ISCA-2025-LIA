import argparse
import dataclasses
import os
import torch
from transformers import OPTForCausalLM, OPTConfig, AutoTokenizer
import numpy as np

@dataclasses.dataclass(frozen=True)
class OptConfig:
    name: str = "opt-175b"
    num_hidden_layers: int = 96
    max_seq_len: int = 2048
    hidden_size: int = 12288
    n_head: int = 96
    input_dim: int = 12288
    ffn_embed_dim: int = 12288 * 4
    pad: int = 1
    activation_fn: str = 'relu'
    vocab_size: int = 50272
    layer_norm_eps: float = 0.00001
    pad_token_id: int = 1
    dtype: type = np.float16

    def model_bytes(self):
        h = self.input_dim
        return  2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def create_opt_model(config):
    model_config = OPTConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_len,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.n_head,
        ffn_dim=config.ffn_embed_dim,
        hidden_act=config.activation_fn,
        layer_norm_eps=config.layer_norm_eps,
        pad_token_id=config.pad_token_id,
        position_embedding_type="absolute"
    )

    # Initialize the model with the configuration
    model = OPTForCausalLM(model_config)

    # Initialize the model weights with dummy values (random)
    for param in model.parameters():
        param.data = torch.rand_like(param.data, dtype=torch.bfloat16)

    return model


def save_model(model, directory):
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=False)
    print(f"Model saved to {directory}")

def save_tokenizer(tokenizer, directory):
    os.makedirs(directory, exist_ok=True)
    tokenizer.save_pretrained(directory)
    print(f"Tokenizer saved to {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt-66b")
    parser.add_argument("--save_dir", type=str, default="/storage/hyungyo2/opt-model/opt-66b/")
    args = parser.parse_args()

    if args.model == "opt-175b":
        config = OptConfig(
            name="opt-175b",
            num_hidden_layers=96,
            max_seq_len=2048,
            hidden_size=12288,
            n_head=96,
            input_dim=12288,
            ffn_embed_dim=49152,
            vocab_size=50272,
            layer_norm_eps=0.00001,
            pad_token_id=1,
            dtype=np.float16
        )

        # Create the OPT model
        OPT = create_opt_model(config)

        # Print the model architecture to confirm
        print(OPT)

        # Save the model
        save_model(OPT, args.save_dir)
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
        # save_tokenizer(tokenizer, args.save_dir)

        # current_vocab_size = len(tokenizer)
        # additional_tokens = 12288 - current_vocab_size
        # if additional_tokens > 0:
        #     tokenizer.add_tokens([f"[DUMMY_{i}]" for i in range(additional_tokens)])
        # Example usage with dummy input
        # input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))  # Random input
        # outputs = OPT(input_ids)
        # print(outputs)

    elif args.model == "opt-66b":
        config = OptConfig(
            name="opt-66b",
            num_hidden_layers=64,
            max_seq_len=2048,
            hidden_size=9216,
            n_head=72,
            input_dim=9216,
            ffn_embed_dim=36864,
            vocab_size=50272,
            layer_norm_eps=0.00001,
            pad_token_id=1,
            dtype=np.float16
        )

        # Create the OPT model
        OPT = create_opt_model(config)

        # Print the model architecture to confirm
        print(OPT)

        # Save the model
        save_model(OPT, args.save_dir)
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
        # save_tokenizer(tokenizer, args.save_dir)

        # current_vocab_size = len(tokenizer)
        # additional_tokens = 12288 - current_vocab_size
        # if additional_tokens > 0:
        #     tokenizer.add_tokens([f"[DUMMY_{i}]" for i in range(additional_tokens)])
        # Example usage with dummy input
        # input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))  # Random input
        # outputs = OPT(input_ids)
        # print(outputs)
