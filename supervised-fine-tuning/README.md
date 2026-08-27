# Supervised Fine-Tuning

Two hardware variants of the same Bielik-11B SFT pipeline:

| Folder | Hardware | Technique | Infrastructure |
|---|---|---|---|
| [`gpu/`](gpu/README.md) | NVIDIA GPU (H100) | QLoRA (4/8-bit via bitsandbytes) | HPC Eagle/Proxima, SLURM |
| [`tpu/`](tpu/README.md) | Google Cloud TPU | bf16 LoRA + FSDPv2/SPMD sharding | TPU VM (Google TPU grant) |

The key difference in one line: **TPUs can't run bitsandbytes/QLoRA (CUDA-only), so the TPU variant trains a bfloat16 LoRA with the model sharded across TPU chips instead of quantized onto one device.**

The TPU port follows the changes validated in the [`trl_on_tpu_working.ipynb`](https://colab.research.google.com/drive/1vcHi8_GcEtgDvJvQxLE87HmBODmV308B) showcase notebook; the full list of differences — and how to smoke-test the TPU pipeline for free, without the grant — is in [`tpu/README.md`](tpu/README.md).
