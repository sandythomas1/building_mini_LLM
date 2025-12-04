import torch
import time

device = "cuda"

# Big matrix size (adjust if you want more load)
N = 20000  

print("Creating large matrices...")
A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)

torch.cuda.synchronize()
print("Starting heavy GPU workload...")

start = time.time()

for i in range(10):
    C = A @ B
    torch.cuda.synchronize()
    print(f"Iteration {i+1}/10 complete")

end = time.time()
print("Done.")
print("Total time:", round(end - start, 2), "seconds")
