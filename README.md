# INT-Crush Converter

Quantize `.safetensors` models to INT4/INT8 with ConvRot rotation + GPTQ/LDLQ/RTN.

> **⚠️ WARNING:** This code has not been thoroughly tested.

*Developed with AI assistance.*

## Usage

```bash
python -m converter.cli \
  -i flux-2-klein-9b.safetensors \
  -o ./quantized \
  --rot-size 256 \
  --quant-method gptq \
  --gptq-block-size 128 \
  -c calibration.pt \
  --exclude-patterns "img_in,time_in,guidance_in,txt_in,double_stream_modulation_img,double_stream_modulation_txt,single_stream_modulation" \
  --int-bits 4
```

Minimal (RTN, no calibration):
```bash
python -m converter.cli -i model.safetensors -o ./out --rot-size 256 --int-bits 4
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | required | Input `.safetensors` |
| `-o, --output` | required | Output directory |
| `--rot-size` | `0` | Hadamard group size: `0`/`16`/`64`/`256` |
| `--int-bits` | `4` | `4` or `8` |
| `--group-size` | `128` | Channels per scale group (INT4 only; INT8 uses one scale per row) |
| `--quant-method` | `rtn` | `rtn` / `gptq` / `ldlq` |
| `-c, --calibration` | — | `.pt` file from ComfyUI-GPTQ-Calibration |
| `--gptq-block-size` | `128` | GPTQ/LDLQ block size |
| `--damping` | `0.01` | GPTQ/LDLQ damping |
| `--ldlq-iterations` | `1` | LDLQ passes |
| `--permuquant` | off | Channel reordering |
| `--tau` | `0.0` | PermuQuant threshold |
| `--exclude-patterns` | — | Extra layers to skip (appended to defaults) |
| `--skip-patterns` | defaults | Replace default skip list entirely |
| `--comfy-compat` | off | ComfyUI-INT8-Fast metadata, INT8 only |

**Default skip patterns:** `embed`, `norm`, `modulation`, `lm_head`, `output`, `proj_out`

## Methods

- **RTN** — round-to-nearest. No calibration. Baseline.
- **GPTQ** — Hessian error compensation. Needs calibration `.pt`. Best quality.
- **LDLQ** — weight-only Hessian (`W^T W / M`). No calibration. Between RTN and GPTQ.

## Rotation (ConvRot)

Regular Hadamard Transform on weight columns before quantization. Suppresses outliers. `--rot-size 0` disables; `16`/`64`/`256` are valid (power of 4). Pads `in_features` to the next multiple of `rot_size`.

Based on [ConvRot (arXiv:2512.03673)](https://arxiv.org/abs/2512.03673).

## PermuQuant

Channel reordering to reduce quantization error. Enabled with `--permuquant`. Uses calibration permutations or weight statistics.

Based on [PermuQuant (arXiv:2605.09503)](https://arxiv.org/abs/2605.09503).

## Output

Single `model.safetensors` containing per-layer tensors:
- `<name>` — `uint8` (INT4, packed 2 per byte) or `int8` (INT8)
- `<name>_scale` — `float16` scales: `[out, num_groups]` (INT4) or `[out, 1]` (INT8)
- `<name>.perm` — optional PermuQuant indices (`int32`)

Metadata: `int_crush.format_version`, `int_crush.method`, `int_crush.rot_size`, `int_crush.packing_order`

## Calibration

GPTQ needs a `.pt` file from [ComfyUI-GPTQ-Calibration](https://github.com/ThunderFun/ComfyUI-GPTQ-Calibration):

1. Load unquantized model in ComfyUI with the calibration node
2. Run images → node collects per-layer Hessians → saves `.pt`
3. Pass to converter: `-c calibration.pt`

Contains:
- `hessians` — full `[in, in]` or block-diagonal `[blocks, bs, bs]`
- `shapes`, `layer_types` — layer metadata

## Notes

- Only 2D weight tensors are quantized.
- `rot_size` must be power of 4 or `0`.
- `group_size` must be power of 2 ≥ 32. INT4 only; INT8 always uses one scale per row.
- GPTQ falls back to RTN for layers without calibration data.


## Tests

```bash
pytest tests/
```
