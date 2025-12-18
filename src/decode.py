import pickle
import os

def decode_output(token_indices, meta_path='data/meta.pkl'):
    if not os.path.exists(meta_path):
        return "Error: meta.pkl not found."
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Use the exact key we found in your file
    int_to_char = meta['int_to_char']
    
    # Convert indices back to text
    decoded_text = "".join([int_to_char[int(i)] for i in token_indices])
    return decoded_text

# Your results from the last terminal run
model_results = [0, 43, 57, 44, 53, 38, 35, 58, 0, 46, 53, 48, 54, 33, 41, 0, 52, 57, 51, 45, 0, 38, 37, 37, 48, 44, 53, 44, 54, 56, 51, 49, 39, 0, 52, 47, 52, 36, 0, 55, 58, 56, 34, 42, 0, 43, 51, 0, 54, 55, 51]

print("--- Decoded Output ---")
print(decode_output(model_results))