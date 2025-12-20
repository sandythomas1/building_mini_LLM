import tiktoken
import numpy as np
import pandas as pd
import os

# 1. Load your data - we'll read it in chunks if it's truly massive
csv_path = '/mnt/c/Users/sandy/Desktop/dev/building_mini_llm/data/customer_support_data.csv'
enc = tiktoken.get_encoding("gpt2")
all_tokens = []

print("Starting tokenization...")

# Read the CSV in chunks of 10,000 rows to save RAM
for chunk in pd.read_csv(csv_path, chunksize=10000):
    # Create the role: text format
    formatted_texts = chunk['role'].astype(str) + ": " + chunk['text'].astype(str) + "\n"
    
    # Encode each chunk and extend our list
    for line in formatted_texts:
        all_tokens.extend(enc.encode_ordinary(line))
    
    print(f"Processed a chunk... current token count: {len(all_tokens)}")

# 2. Convert to numpy and save
print("Finalizing binary file...")
tokens_np = np.array(all_tokens, dtype=np.uint16)
tokens_np.tofile('data/training_data_v2.bin')

# 3. Save metadata
import pickle
meta = {'vocab_size': 50257} 
with open('data/meta_v2.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f"Success! Total tokens: {len(all_tokens)}")