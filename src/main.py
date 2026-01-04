import torch
from data_loader import Dataloader
from model import Transformer, TransformerConfig
import time

# 1. Setup Hyperparameters
config = TransformerConfig(
    vocab_size=50257,     #50257,    # GPT-2 vocabulary size
    block_size=128,   # context length
    n_layer=8, 
    n_head=8, 
    n_embd=768
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.1)

train_data_path = "/mnt/c/Users/sandy/Desktop/dev/building_mini_llm/data/training_data.bin" # Path to training data
train_loader = Dataloader(train_data_path, batch_size=16, block_size=config.block_size)

@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=20):
    out = {}
    model.eval() # Set to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set back to training mode
    return out

# 2. Example Training Loop
def train(model, optimizer, loader, device, start_step=0, total_steps=12500):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.1)

    startTime = time.time()
    print(f"Training started at {time.ctime(startTime)}")

    for i in range(start_step, total_steps):
        
        # Every 250 steps, check how we are doing on validation data
        if i % 250 == 0:
            losses = estimate_loss(model, loader, device)
            print(f"Step {i}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            
            # Save checkpoint if it's a major milestone
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'loss': losses['val'],
            }
            torch.save(checkpoint, 'model_checkpoint.pt')

        # Standard training step
        xb, yb = loader.get_batch('train', device)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(xb, yb)
        #optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    endTime = time.time()
    totalDuration = endTime - startTime

    print("Training complete")
    if totalDuration < 3600:
        print(f"Total time: {totalDuration / 60:.2f} minutes")
    else:
        print(f"Total time: {totalDuration / 3600:.2f} hours")

if __name__ == "__main__":
    train(model, optimizer, train_loader, device)
    
    # After training, try generating again!
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=50)
    print("Generated Tokens:", generated[0].tolist())