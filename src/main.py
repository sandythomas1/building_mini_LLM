import torch
from data_loader import Dataloader
from model import Transformer, TransformerConfig

# 1. Setup Hyperparameters
config = TransformerConfig(
    vocab_size=50257,    # GPT-2 vocabulary size
    block_size=256, 
    n_layer=6, 
    n_head=6, 
    n_embd=384
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_data_path = "/mnt/c/Users/sandy/Desktop/dev/building_mini_llm/data/training_data.bin" # Path to training data
train_loader = Dataloader(train_data_path, batch_size=32, block_size=config.block_size)

# 2. Example Training Loop
def train(model, optimizer, steps=500):
    model.train()
    for i in range(steps):
        # Get REAL data from the loader
        xb, yb = train_loader.get_batch(device)

        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

if __name__ == "__main__":
    train(model, optimizer)
    
    # After training, try generating again!
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=50)
    print("Generated Tokens:", generated[0].tolist())