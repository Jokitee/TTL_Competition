"""
migrate_weights.py
migrate old (16->64->48->32->7) weights to v5 branched architecture
"""
import pickle, os, sys
import torch
sys.path.insert(0, "E:\\Test_FIre")
from train_ppo import ActorCritic

OLD_PATH = "E:\\Test_FIre\\best_model_ppo_dynamic_attacker.pkl"
NEW_PATH = "E:\\Test_FIre\\best_model_v5_branched.pkl"

if not os.path.exists(OLD_PATH):
    print(f"[ERROR] Old model not found: {OLD_PATH}")
    exit(1)

if os.path.exists(NEW_PATH):
    print(f"[INFO] v5 model already exists: {NEW_PATH}")
    print("[INFO] Delete it manually to re-migrate.")
    exit(0)

with open(OLD_PATH, "rb") as f:
    w = pickle.load(f)
print(f"[OK] Loaded old model: {len(w)} weight arrays")
for i, wi in enumerate(w):
    print(f"     w[{i}] shape={wi.shape}")

device = "cpu"
model = ActorCritic().to(device)

with torch.no_grad():
    # shared 16->64: perfect match with old w[0]/w[1]
    model.shared[0].weight.copy_(torch.FloatTensor(w[0].T))
    model.shared[0].bias.copy_(torch.FloatTensor(w[1]))
    print("[OK] shared layer migrated (16->64)")

    # offense_branch 64->48: perfect match with old w[2]/w[3]
    model.offense_branch[0].weight.copy_(torch.FloatTensor(w[2].T))
    model.offense_branch[0].bias.copy_(torch.FloatTensor(w[3]))
    print("[OK] offense_branch migrated (64->48)")

    # tactical_branch 64->48: same source, will diverge during training
    model.tactical_branch[0].weight.copy_(torch.FloatTensor(w[2].T))
    model.tactical_branch[0].bias.copy_(torch.FloatTensor(w[3]))
    print("[OK] tactical_branch migrated (64->48)")

    # output heads: dimension mismatch (old: 32->?, new: 48->?), keep random init
    print("[--] Output heads (mv/rt/fire) kept as random init (dim mismatch: 32 vs 48)")

weights = model.export_weights()
with open(NEW_PATH, "wb") as f:
    pickle.dump(weights, f)

size_kb = os.path.getsize(NEW_PATH) / 1024
print(f"\n[DONE] Migration complete!")
print(f"       Output: {NEW_PATH} ({size_kb:.1f} KB)")
print(f"       Shared perception + branch layers inherited from 7M-step training.")
print(f"       Output heads will re-adapt within ~100k steps.")
print(f"\nRun: python train_ppo.py")
