import torch
from torch import nn
from functools import partial
from models_audio_mae import AudioMaskedAutoencoderViT

audio_mels = torch.ones([2, 1, 1024, 128])

# Paper recommended archs
model = AudioMaskedAutoencoderViT(
    num_mels=128, mel_len=1024, in_chans=1,
    patch_size=16, embed_dim=768, encoder_depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

loss, pred, mask = model(audio_mels)
print(f"loss : {loss}")