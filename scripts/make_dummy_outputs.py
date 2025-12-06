import os
import torch

# Make sure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Toy sizes – small but realistic-ish
L = 80   # number of residues
S = 16   # number of MSA sequences
C_m = 64 # MSA / single embedding dim
C_z = 32 # pair embedding dim

msa = torch.randn(1, S, L, C_m)      # [*, N_seq, N_res, C_m]
pair = torch.randn(1, L, L, C_z)     # [*, N_res, N_res, C_z]
single = torch.randn(1, L, C_m)      # [*, N_res, C_s]

outputs = {
    "msa": msa,
    "pair": pair,
    "single": single,
}

out_path = "outputs/6KWC_outputs.pt"
torch.save(outputs, out_path)
print(f"Saved dummy outputs to {out_path}")


