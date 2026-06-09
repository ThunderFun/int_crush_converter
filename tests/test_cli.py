"""Tests for quant.cli — CLI utilities and quantization pipeline."""

from converter.cli import should_skip, DEFAULT_SKIP_PATTERNS


class TestShouldSkip:
    def test_embed_skipped(self):
        assert should_skip("model.embed_tokens.weight", DEFAULT_SKIP_PATTERNS)

    def test_norm_skipped(self):
        assert should_skip("model.layer_norm.weight", DEFAULT_SKIP_PATTERNS)
        assert should_skip("model.rmsnorm.weight", DEFAULT_SKIP_PATTERNS)

    def test_modulation_skipped(self):
        assert should_skip("model.modulation.weight", DEFAULT_SKIP_PATTERNS)

    def test_lm_head_skipped(self):
        assert should_skip("lm_head.weight", DEFAULT_SKIP_PATTERNS)

    def test_output_skipped(self):
        assert should_skip("model.output.weight", DEFAULT_SKIP_PATTERNS)

    def test_proj_out_skipped(self):
        assert should_skip("model.proj_out.weight", DEFAULT_SKIP_PATTERNS)

    def test_linear_not_skipped(self):
        assert not should_skip("model.layers.0.self_attn.q_proj.weight", DEFAULT_SKIP_PATTERNS)

    def test_case_insensitive(self):
        assert should_skip("model.Embed.weight", DEFAULT_SKIP_PATTERNS)
        assert should_skip("model.NORM.weight", DEFAULT_SKIP_PATTERNS)

    def test_custom_patterns(self):
        patterns = ["custom", "special"]
        assert should_skip("model.custom_layer.weight", patterns)
        assert not should_skip("model.normal_layer.weight", patterns)

    def test_empty_patterns(self):
        assert not should_skip("model.embed.weight", [])

    def test_partial_match(self):
        """Pattern matching is substring-based."""
        assert should_skip("embedding.weight", DEFAULT_SKIP_PATTERNS)
        assert should_skip("normalization.weight", DEFAULT_SKIP_PATTERNS)
