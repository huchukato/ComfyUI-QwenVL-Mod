"""
SageAttention patching functions for QwenVL models.
This module contains the safe SageAttention implementation with proper error handling.
"""

import torch


def set_sage_attention(model):
    """
    Apply SageAttention patching to the model.
    Patches Qwen2Attention and Qwen3VLTextAttention modules to use SageAttention kernels.
    """
    from AILab_QwenVL import sage_attn_available, get_sage_attention_config
    
    if not sage_attn_available():
        raise ImportError("SageAttention library is not installed or GPU doesn't support it.")

    SAGE_ATTN_FUNC, QK_QUANT_GRAN, PV_ACCUM_DTYPE = get_sage_attention_config()
    if SAGE_ATTN_FUNC is None:
        raise RuntimeError("No compatible SageAttention kernel found for this GPU.")

    # Try to import different attention classes for different Qwen models
    attention_classes = []

    # Qwen2 models
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb as qwen2_apply_rotary
        attention_classes.append((Qwen2Attention, qwen2_apply_rotary))
    except ImportError:
        pass

    # Qwen3 models (Qwen3-VL, etc.)
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb as qwen3_apply_rotary
        attention_classes.append((Qwen3Attention, qwen3_apply_rotary))
    except ImportError:
        pass

    # Qwen3-VL specific
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, apply_rotary_pos_emb as qwen3vl_apply_rotary
        attention_classes.append((Qwen3VLTextAttention, qwen3vl_apply_rotary))
    except ImportError:
        pass

    if not attention_classes:
        print("[QwenVL] Could not import any attention classes for SageAttention patching")
        return

    def make_sage_forward(AttentionClass, apply_rotary_pos_emb_func):
        def sage_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple = None,
            attention_mask: torch.Tensor = None,
            past_key_values=None,
            cache_position: torch.LongTensor = None,
            position_ids: torch.LongTensor = None,
            **kwargs,
        ):
            original_dtype = hidden_states.dtype

            # Determine target dtype
            is_4bit = hasattr(self.q_proj, 'quant_state')
            if is_4bit:
                target_dtype = torch.bfloat16
            else:
                target_dtype = self.q_proj.weight.dtype

            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            bsz, q_len = input_shape[0], input_shape[1] if len(input_shape) > 1 else hidden_states.size(1)

            # Handle q_norm and k_norm for Qwen3-VL
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Apply normalization if available (Qwen3-VL specific)
            if hasattr(self, 'q_norm'):
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            else:
                query_states = query_states.view(hidden_shape).transpose(1, 2)

            if hasattr(self, 'k_norm'):
                key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            else:
                key_states = key_states.view(hidden_shape).transpose(1, 2)

            value_states = value_states.view(hidden_shape).transpose(1, 2)

            # Apply rotary embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_func(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin if position_embeddings else None, "cos": cos if position_embeddings else None, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            is_causal = attention_mask is None and q_len > 1

            attn_output = SAGE_ATTN_FUNC(
                query_states.to(target_dtype),
                key_states.to(target_dtype),
                value_states.to(target_dtype),
                tensor_layout="HND",
                is_causal=is_causal,
                qk_quant_gran=QK_QUANT_GRAN,
                pv_accum_dtype=PV_ACCUM_DTYPE,
            )

            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)

            attn_output = self.o_proj(attn_output)

            if attn_output.dtype != original_dtype:
                attn_output = attn_output.to(original_dtype)

            return attn_output, None

        return sage_attention_forward

    # Apply patching to all supported attention modules
    patched_count = 0
    for AttentionClass, apply_rotary_func in attention_classes:
        sage_forward = make_sage_forward(AttentionClass, apply_rotary_func)
        for module in model.modules():
            if isinstance(module, AttentionClass):
                module.forward = sage_forward.__get__(module, AttentionClass)
                patched_count += 1

    if patched_count > 0:
        print(f"[QwenVL] SageAttention: Patched {patched_count} attention layers")
    else:
        print("[QwenVL] SageAttention: No compatible attention layers found to patch")
