"""Only Qwen3.5 model architectures are advertised by this release."""

def test_registry_exposes_only_qwen35_dense_and_moe():
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
    assert set(MODEL_ARCH_MAPPING) == {
        'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration',
        'Qwen3_5MoeForCausalLM', 'Qwen3_5MoeForConditionalGeneration',
    }
