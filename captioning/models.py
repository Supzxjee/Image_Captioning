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
    def __init__(self, embed_dim, num_heads, num_queries=32, num_layers=2, dropout=0.1,
                 prompt_conditioned=False):
        super().__init__()
        self.prompt_conditioned = prompt_conditioned
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        if prompt_conditioned:
            self.prompt_condition_projection = nn.Linear(embed_dim, embed_dim)
            self.prompt_condition_gate = nn.Linear(embed_dim * 2, embed_dim)
            self.prompt_condition_norm = nn.LayerNorm(embed_dim)
            nn.init.zeros_(self.prompt_condition_gate.weight)
            nn.init.constant_(self.prompt_condition_gate.bias, -2.0)
        self.layers = nn.ModuleList([
            VisualQueryLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, visual_tokens, prompt_tokens=None, prompt_mask=None):
        queries = self.query_tokens.expand(visual_tokens.size(0), -1, -1)
        if self.prompt_conditioned:
            if prompt_tokens is None or prompt_mask is None:
                raise ValueError('Prompt-conditioned Q-Former requires prompt tokens and mask.')
            valid = prompt_mask.to(dtype=prompt_tokens.dtype).unsqueeze(-1)
            pooled_prompt = (prompt_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            condition = self.prompt_condition_projection(pooled_prompt).unsqueeze(1)
            condition = condition.expand(-1, queries.size(1), -1)
            gate = torch.sigmoid(self.prompt_condition_gate(
                torch.cat([queries, condition], dim=-1)
            ))
            queries = self.prompt_condition_norm(queries + gate * condition)
        for layer in self.layers:
            queries = layer(queries, visual_tokens)
        return queries


class ObjectSemanticAlignment(nn.Module):
    """Retrieve object-prompt evidence for each visual query and fuse it through a gate."""
    def __init__(self, embed_dim, num_heads, dropout=0.1, object_dim=512):
        super().__init__()
        self.object_projection = nn.Sequential(
            nn.Linear(object_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_queries, object_tokens, object_mask):
        if object_tokens.ndim != 3 or object_mask.ndim != 2:
            raise ValueError('Expected object tokens (B, O, 512) and mask (B, O).')
        if object_tokens.shape[:2] != object_mask.shape:
            raise ValueError('Object token and mask dimensions do not match.')
        valid = object_mask.to(dtype=torch.bool)
        has_objects = valid.any(dim=1)

        # MultiheadAttention cannot consume a row whose every key is masked.
        safe_valid = valid.clone()
        safe_tokens = object_tokens
        if (~has_objects).any():
            safe_valid[~has_objects, 0] = True
            safe_tokens = object_tokens.clone()
            safe_tokens[~has_objects, 0] = 0

        object_features = self.object_projection(
            safe_tokens.to(dtype=self.object_projection[0].weight.dtype))
        attended, _ = self.cross_attn(
            query=visual_queries,
            key=object_features,
            value=object_features,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate(torch.cat([visual_queries, attended], dim=-1)))
        aligned = visual_queries + self.dropout(gate * attended)
        # An image with no accepted detections must preserve the baseline exactly.
        return torch.where(has_objects[:, None, None], aligned, visual_queries)


class CascadeSemanticAlignment(nn.Module):
    """Use object context to select object-relation prompt tokens for visual queries."""
    def __init__(self, embed_dim, num_heads, dropout=0.1, object_dim=512):
        super().__init__()
        self.object_projection = nn.Sequential(
            nn.Linear(object_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.object_context = nn.Linear(embed_dim, embed_dim)
        self.prompt_key = nn.Linear(embed_dim, embed_dim)
        self.selector = nn.Linear(embed_dim, 1)
        self.query_to_semantic = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_queries, prompt_features, prompt_mask,
                object_tokens, object_mask):
        object_valid = object_mask.to(dtype=torch.bool)
        prompt_valid = prompt_mask.to(dtype=torch.bool)
        has_context = object_valid.any(dim=1) & prompt_valid.any(dim=1)

        projected_objects = self.object_projection(
            object_tokens.to(dtype=self.object_projection[0].weight.dtype))
        object_weights = object_valid.to(projected_objects.dtype).unsqueeze(-1)
        pooled_objects = ((projected_objects * object_weights).sum(dim=1) /
                          object_weights.sum(dim=1).clamp_min(1))
        selector_hidden = torch.tanh(
            self.prompt_key(prompt_features) +
            self.object_context(pooled_objects).unsqueeze(1))
        scores = self.selector(selector_hidden).squeeze(-1)

        safe_prompt_valid = prompt_valid.clone()
        if (~has_context).any():
            safe_prompt_valid[~has_context, 0] = True
        scores = scores.masked_fill(~safe_prompt_valid, torch.finfo(scores.dtype).min)
        selection = torch.softmax(scores, dim=1)
        # Retain the original feature scale while emphasizing selected tokens.
        selection = selection * prompt_valid.sum(dim=1, keepdim=True).clamp_min(1)
        selected_prompt = prompt_features * selection.unsqueeze(-1)
        if (~has_context).any():
            selected_prompt = selected_prompt.clone()
            selected_prompt[~has_context, 0] = 0

        attended, _ = self.query_to_semantic(
            query=visual_queries,
            key=selected_prompt,
            value=selected_prompt,
            key_padding_mask=~safe_prompt_valid,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate(torch.cat([visual_queries, attended], dim=-1)))
        aligned = visual_queries + self.dropout(gate * attended)
        return torch.where(has_context[:, None, None], aligned, visual_queries)


class UniversalVisionEncoder(nn.Module):
    def __init__(self, model_name='clip', embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 attn_dropout=0.1, visual_precision='fp32', load_backbone=True,
                 visual_adapter='direct', num_visual_queries=32, qformer_layers=2,
                 use_itc=False, prompt_conditioned_qformer=False,
                 object_semantic_alignment=False, cascade_semantic_alignment=False):
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
        self.prompt_conditioned_qformer = prompt_conditioned_qformer
        self.object_semantic_alignment = object_semantic_alignment
        self.cascade_semantic_alignment = cascade_semantic_alignment
        self.use_itc = use_itc
        self.vis_projection = nn.Linear(768, embed_dim)
        self.qformer = (LightweightQFormer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_queries=num_visual_queries,
            num_layers=qformer_layers,
            dropout=attn_dropout,
            prompt_conditioned=prompt_conditioned_qformer,
        ) if visual_adapter == 'qformer' else None)
        self.object_alignment = (ObjectSemanticAlignment(
            embed_dim, num_heads, attn_dropout
        ) if object_semantic_alignment else None)
        self.cascade_alignment = (CascadeSemanticAlignment(
            embed_dim, num_heads, attn_dropout
        ) if cascade_semantic_alignment else None)
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

    def forward(self, images, cached_prompt_tokens, prompt_mask,
                object_prompt_tokens=None, object_prompt_mask=None):
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
        if self.qformer is not None and self.prompt_conditioned_qformer:
            prompt_features = self.prompt_projection(cached_prompt_tokens)
            visual_memory = self.qformer(vis_features, prompt_features, prompt_mask)
        else:
            visual_memory = self.qformer(vis_features) if self.qformer is not None else vis_features
            # Preserve dropout/RNG order for every pre-existing experiment.
            prompt_features = self.prompt_projection(cached_prompt_tokens)

        if self.object_alignment is not None:
            if object_prompt_tokens is None or object_prompt_mask is None:
                raise ValueError('Object semantic alignment requires object prompt tokens and mask.')
            visual_memory = self.object_alignment(
                visual_memory, object_prompt_tokens, object_prompt_mask)
        if self.cascade_alignment is not None:
            if object_prompt_tokens is None or object_prompt_mask is None:
                raise ValueError('Cascade semantic alignment requires object prompt tokens and mask.')
            visual_memory = self.cascade_alignment(
                visual_memory, prompt_features, prompt_mask,
                object_prompt_tokens, object_prompt_mask)

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

    def build_memory(self, images, cached_prompt_tokens, prompt_mask,
                     object_prompt_tokens=None, object_prompt_mask=None):
        vis_features, grounded_prompt_features = self.encoder(
            images, cached_prompt_tokens, prompt_mask,
            object_prompt_tokens, object_prompt_mask,
        )
        memory = torch.cat([grounded_prompt_features, vis_features], dim=1)
        prompt_pad_mask = (prompt_mask == 0)
        visual_pad_mask = torch.zeros(
            images.size(0), vis_features.size(1), dtype=torch.bool, device=images.device
        )
        memory_pad_mask = torch.cat([prompt_pad_mask, visual_pad_mask], dim=1)
        return memory, memory_pad_mask

    def forward(self, images, cached_prompt_tokens, prompt_mask, tgt, tgt_key_padding_mask=None,
                return_visual=False, return_itc=False, object_prompt_tokens=None,
                object_prompt_mask=None):
        memory, memory_pad_mask = self.build_memory(
            images, cached_prompt_tokens, prompt_mask,
            object_prompt_tokens, object_prompt_mask)
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
