import torch.nn as nn
import re
from .perceiver import PerceiverResampler


class IdentityMap(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, *args, **kwargs):
		return x

	@property
	def config(self):
		return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.pre_norm = nn.LayerNorm(channels)

		self.proj = nn.Sequential(
			nn.Linear(channels, channels),
			nn.GELU(),
			nn.Linear(channels, channels)
		)
	def forward(self, x):
		x = self.pre_norm(x)
		return x + self.proj(x)


def build_vision_projector(config, object_projector=False, delay_load=False, **kwargs):
	if not object_projector:
		projector_type = getattr(config, 'mm_projector_type', 'linear')
	else:
		projector_type = getattr(config, 'object_mm_projector_type', 'perceiver')
	
	if projector_type == 'linear':
		return nn.Linear(config.mm_hidden_size, config.hidden_size)

	mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
	if mlp_gelu_match:
		mlp_depth = int(mlp_gelu_match.group(1))
		modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
		for _ in range(1, mlp_depth):
			modules.append(nn.GELU())
			modules.append(nn.Linear(config.hidden_size, config.hidden_size))
		return nn.Sequential(*modules)

	if projector_type == 'identity':
		return IdentityMap()

	if projector_type == "perceiver":
		return nn.Sequential(
					nn.LayerNorm(config.mm_hidden_size),
					PerceiverResampler(
					dim = config.mm_hidden_size,
					dim_head = 96,
					depth = 6,
					heads = 16,
					num_latents = 32,
					num_media_embeds = 1
					),
					nn.Linear(
					config.mm_hidden_size, config.hidden_size
					)
					)

	raise ValueError(f'Unknown projector type: {projector_type}')
