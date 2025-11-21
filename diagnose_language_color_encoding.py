"""Test if BERT language encoder captures color information."""
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Test instructions
instructions = [
    "Pick up the red sphere and place it in the goal bin.",
    "Pick up the green sphere and place it in the goal bin.",
    "Pick up the blue sphere and place it in the goal bin.",
    "Pick up the yellow sphere and place it in the goal bin.",
    "Pick up the purple sphere and place it in the goal bin.",
]

print("="*70)
print("LANGUAGE ENCODER COLOR DISCRIMINATION TEST")
print("="*70)

embeddings = []
with torch.no_grad():
    for instruction in instructions:
        inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :].squeeze()  # CLS token
        embeddings.append(cls_embed.numpy())
        
        color = instruction.split()[3]  # Extract color
        print(f"\n{color:6s}: CLS embedding norm = {torch.norm(cls_embed).item():.4f}")

# Compute pairwise cosine similarities
embeddings = np.array(embeddings)
print("\n" + "="*70)
print("PAIRWISE COSINE SIMILARITIES")
print("="*70)
print("\nIf BERT captures color, different colors should have LOWER similarity.")
print("If all similarities are ~0.99, BERT treats all instructions as identical!\n")

# Compute cosine similarity manually
def cosine_similarity_matrix(X):
    # Normalize rows
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Compute dot product
    return X_norm @ X_norm.T

sim_matrix = cosine_similarity_matrix(embeddings)

colors = ["red", "green", "blue", "yellow", "purple"]
print(f"{'':8s}", end="")
for color in colors:
    print(f"{color:8s}", end="")
print()

for i, color1 in enumerate(colors):
    print(f"{color1:8s}", end="")
    for j, color2 in enumerate(colors):
        print(f"{sim_matrix[i, j]:8.4f}", end="")
    print()

# Compute average similarity for same vs different colors
same_color = sim_matrix[np.diag_indices_from(sim_matrix)]
diff_color_mask = ~np.eye(len(colors), dtype=bool)
diff_color = sim_matrix[diff_color_mask]

print(f"\n{'='*70}")
print(f"Same color similarity (diagonal):  {same_color.mean():.6f}")
print(f"Diff color similarity (off-diag): {diff_color.mean():.6f}")
print(f"Difference:                         {same_color.mean() - diff_color.mean():.6f}")

if diff_color.mean() > 0.995:
    print(f"\n❌ PROBLEM: BERT embeddings are TOO SIMILAR (>{0.995:.3f})")
    print(f"   The language encoder is NOT capturing color information!")
    print(f"   All instructions look nearly identical to the model.")
elif diff_color.mean() > 0.98:
    print(f"\n⚠️  WARNING: BERT embeddings are VERY SIMILAR (>{0.98:.3f})")
    print(f"   The color signal is WEAK. Model may struggle to distinguish.")
else:
    print(f"\n✓ BERT embeddings capture color differences (similarity < 0.98)")
print("="*70)

