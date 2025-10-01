#!/usr/bin/env -S uv run
import torch 
# model_path = './train_dir/osva_sept_8_final_epoch_15730_base_0.7_0.5_1.0_0.8/osva_sept_8_final_epoch_15730_base_0.7_0.5_1.0_0.8/checkpoints/osva_sept_8_final_epoch_15730_base_0.7_0.5_1.0_0.8__e15730__s15402816000__t49653__sc0.pt'
model_path = './train_dir/osva_sept_8_final_epoch_15730_ablationsrandom_0.7_0.5_1.0_0.8/osva_sept_8_final_epoch_15730_ablationsrandom_0.7_0.5_1.0_0.8/checkpoints/osva_sept_8_final_epoch_15730_ablationsrandom_0.7_0.5_1.0_0.8__e15770__s15441984000__t56094__sc0.pt'
model = torch.load(model_path , weights_only=False)

# print(model)
# print(sum(p.numel() for p in model.parameters()))

for n, p in model.named_parameters():
    print(n, p.shape)


