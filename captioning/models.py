"""CLIP visual encoder, gated prompt attention and caption decoder."""
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
from transformers import CLIPModel
from .config import EMBED_DIM, NUM_HEADS


class VisualQueryLayer(nn.Module):
    """Self-attend queries, retrieve visual evidence, then refine with an FFN."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, queries, visual_tokens):
        update, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.self_norm(queries + self.dropout(update))
        update, _ = self.cross_attn(
            query=queries, key=visual_tokens, value=visual_tokens, need_weights=False)
        queries = self.cross_norm(queries + self.dropout(update))
        return self.ffn_norm(queries + self.dropout(self.ffn(queries)))


class LightweightQFormer(nn.Module):
    """Trainable visual queries inspired by BLIP-2 Q-Former, trained from scratch."""
    def __init__(self, embed_dim, num_heads, num_queries=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            VisualQueryLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, visual_tokens):
        queries = self.query_tokens.expand(visual_tokens.size(0), -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual_tokens)
        return queries


class UniversalVisionEncoder(nn.Module):
    def __init__(self, model_name='clip', embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 attn_dropout=0.1, visual_precision='fp32', load_backbone=True,
                 visual_adapter='direct', num_visual_queries=32, qformer_layers=2,
                 use_itc=False):
        super().__init__()
        if model_name.lower() != 'clip':
            raise NotImplementedError('Only CLIP is supported in this notebook.')

        self.feature_extractor = None
        if load_backbone:
            clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
            self.feature_extractor = clip_model.vision_model
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        self.visual_precision = visual_precision
        self.visual_adapter = visual_adapter
        self.num_visual_queries = num_visual_queries
        self.qformer_layers = qformer_layers
        self.use_itc = use_itc
        self.vis_projection = nn.Linear(768, embed_dim)
        self.qformer = (LightweightQFormer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_queries=num_visual_queries,
            num_layers=qformer_layers,
            dropout=attn_dropout,
        ) if visual_adapter == 'qformer' else None)
        self.itc_query_projection = (nn.Linear(embed_dim, 512) if use_itc else None)
        self.prompt_projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
        )

        # H1.2: Q = prompt tokens, K = V = visual tokens.
        self.prompt_to_visual_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        # Gate sees both the original prompt and retrieved visual information.
        # Initial sigmoid(-2) ~= 0.119: start with a small visual update.
        self.prompt_visual_gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.prompt_visual_gate.weight)
        nn.init.constant_(self.prompt_visual_gate.bias, -2.0)
        self.prompt_attn_norm = nn.LayerNorm(embed_dim)
        self.prompt_attn_dropout = nn.Dropout(attn_dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def train(self, mode=True):
        super().train(mode)
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
        return self

    def extract_visual_features(self, images):
        if self.feature_extractor is None:
            raise RuntimeError('CLIP vision backbone was not loaded; provide cached visual tokens.')
        use_amp = self.visual_precision == 'amp-fp16' and images.device.type == 'cuda'
        context = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
        with torch.no_grad(), context:
            features = self.feature_extractor(pixel_values=images).last_hidden_state
        return features.to(dtype=self.vis_projection.weight.dtype)

    def forward(self, images, cached_prompt_tokens, prompt_mask):
        if images.ndim == 3:
            # Cached raw CLIP last_hidden_state; no backbone forward pass.
            if images.shape[1:] != (197, 768):
                raise ValueError(f'Expected cached visual tokens (B, 197, 768), got {images.shape}')
            visual_features = images.to(dtype=self.vis_projection.weight.dtype)
        elif images.ndim == 4:
            visual_features = self.extract_visual_features(images)
        else:
            raise ValueError('Expected image pixels (B, C, H, W) or cached tokens (B, 197, 768).')

        vis_features = self.dropout(self.relu(self.vis_projection(visual_features)))
        visual_memory = self.qformer(vis_features) if self.qformer is not None else vis_features
        prompt_features = self.prompt_projection(cached_prompt_tokens)

        attended_prompt, _ = self.prompt_to_visual_attn(
            query=prompt_features,
            key=visual_memory,
            value=visual_memory,
            need_weights=False,
        )

        gate = torch.sigmoid(self.prompt_visual_gate(
            torch.cat([prompt_features, attended_prompt], dim=-1)
        ))
        grounded_prompt_features = self.prompt_attn_norm(
            prompt_features + self.prompt_attn_dropout(gate * attended_prompt)
        )

        prompt_pad_mask = (prompt_mask == 0)
        grounded_prompt_features = grounded_prompt_features.masked_fill(
            prompt_pad_mask.unsqueeze(-1),
            0.0,
        )

        return visual_memory, grounded_prompt_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMBED_DIM, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=NUM_HEADS, num_layers=3, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    @staticmethod
    def causal_mask(size, device):
        # Boolean mask matches the boolean padding masks expected by PyTorch.
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_embeddings = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        return self.fc_out(self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=self.causal_mask(tgt.size(1), tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ))

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def build_memory(self, images, cached_prompt_tokens, prompt_mask):
        vis_features, grounded_prompt_features = self.encoder(
            images, cached_prompt_tokens, prompt_mask
        )
        memory = torch.cat([grounded_prompt_features, vis_features], dim=1)
        prompt_pad_mask = (prompt_mask == 0)
        visual_pad_mask = torch.zeros(
            images.size(0), vis_features.size(1), dtype=torch.bool, device=images.device
        )
        memory_pad_mask = torch.cat([prompt_pad_mask, visual_pad_mask], dim=1)
        return memory, memory_pad_mask

    def forward(self, images, cached_prompt_tokens, prompt_mask, tgt, tgt_key_padding_mask=None,
                return_visual=False, return_itc=False):
        memory, memory_pad_mask = self.build_memory(images, cached_prompt_tokens, prompt_mask)
        vis_features = memory[:, prompt_mask.size(1):]
        batch_size, memory_len, embed_dim = memory.shape
        captions_per_image = tgt.size(0) // batch_size
        if tgt.size(0) != batch_size * captions_per_image:
            raise ValueError('Target batch size must be a multiple of image batch size.')

        memory = memory.unsqueeze(1).expand(batch_size, captions_per_image, memory_len, embed_dim)
        memory = memory.reshape(batch_size * captions_per_image, memory_len, embed_dim)
        memory_pad_mask = memory_pad_mask.unsqueeze(1).expand(batch_size, captions_per_image, memory_len)
        memory_pad_mask = memory_pad_mask.reshape(batch_size * captions_per_image, memory_len)

        logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        if return_itc:
            if self.encoder.itc_query_projection is None:
                raise RuntimeError('ITC output requested, but the ITC projection is disabled.')
            # Keep this trainable projection inside DDP's forward graph so its
            # gradients are synchronized by Accelerate on every process.
            return logits, self.encoder.itc_query_projection(vis_features)
        return (logits, vis_features) if return_visual else logits
