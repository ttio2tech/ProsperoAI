# Supported models

The Models screen can download each curated preset below. These presets use the runtimes and
adapters already included in ProsperoAI; adding one does not require a model-specific code
change. Presets are pinned to revisions and checked against their expected file sizes and
SHA-256 hashes during download.

| Model | Type | Download | PS5 status |
| --- | --- | --- | --- |
| [Mistral 7B Instruct v0.3 Q4_0](https://huggingface.co/QuantFactory/Mistral-7B-Instruct-v0.3-GGUF) | Text, GGUF | Curated preset | Verified; three correct benchmark runs |
| [Qwen3.8 9B Distill Q4_K_M](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) | Text, GGUF | Curated preset | Verified; three correct benchmark runs |
| [MiniCPM5 2B Q8_0](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) | Text, GGUF | Curated preset | Verified, including generation at the 32K context boundary |
| [Ternary Bonsai 2 27B PTQ1_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | Text, GGUF | Curated preset | Verified; three correct benchmark runs |
| [Granite 4.0 H-Tiny Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF) | Text, GGUF | Curated preset | Verified; three correct benchmark runs |
| [BoHF 12B MoE 3A Q3_K_M](https://huggingface.co/theprint/Bohf-12B-MoE-3A-GGUF) | Text, GGUF | Curated preset | Verified; experimental model |
| [SD-Turbo FP16](https://huggingface.co/blackbearreloaded/ProsperoAI-SD-Turbo-FP16-PS5) | Image | Curated preset | Verified; three valid 512 × 512 images |
| [Realistic Vision v5.1 FP16](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) | Image | Curated preset | Verified; three valid 512 × 512 images |
| [Stable Audio 3 Small SFX Q8_0](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Stable-Audio-3-Small-SFX-GGUF) | Sound effects | Curated preset | Verified; three valid stereo WAV generations |
| [ACE-Step 1.5 Turbo Q8_0](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/ACE-Step1.5-GGUF/turbo) | Music | Curated preset | Verified; three valid stereo WAV generations |
| [Kokoro 82M Vulkan](https://huggingface.co/blackbearreloaded/ProsperoAI-Kokoro-82M-FP16-PS5) | Speech | Curated preset | Verified; three valid WAV generations |
| [Chatterbox 500M Q8_0](https://huggingface.co/audio-cpp/audio.cpp-gguf) | Speech, voice cloning | Curated preset | Verified; three valid WAV generations; 19 selectable languages |

The table is the complete list of curated presets in this build. Curated media presets include
their required model files and use the image, audio, or speech runtime already in the app.

## Pending: Qwen Image 2.1

The image adapter recognizes a Qwen Image 2.1 folder that holds `qwen_image_2.1-Q4_K.gguf`
(7B diffusion transformer), `Qwen3VL-8B-Instruct-Q4_K_M.gguf` (text encoder) and
`qwen_image_2.1_vae_bf16.safetensors`. It is not yet a curated preset: its revisions, sizes and
SHA-256 values are not pinned and it has not run on a PS5.

- To test before pinning, copy the three files into one folder such as
  `/data/prosperoai/models/qwen-image-2.1-q4-k/`. The app lists it as
  "qwen-image-2.1-q4-k (Qwen Image 2.1)", or "… (Qwen Image 2.1 Turbo)" with the Turbo file.
- Pin it from a host with Hugging Face access, then review the diff:
  `python3 tools/pin-pending-preset.py model-tools/pending-presets/qwen-image-2.1-q4-k.json`
- The encoder (about 4.7 GiB) and the diffusion weights (about 3.9 GiB) do not fit together
  with working buffers in the 11.4 GiB Vulkan heap. The adapter keeps the encoder on
  disk residency, so stable-diffusion.cpp loads it for the prompt and evicts it before the
  diffusion weights load, and it releases the context after every image.
- Settings: 768 × 768, 20 Euler steps, CFG 6.0, engine-selected flow schedule.
- Turbo: put the [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
  diffusion weights, quantized to Q4_K, in the folder as `qwen_image_2.1_turbo-Q4_K.gguf`
  (it takes precedence over the base file). The adapter then uses upstream sd-cli's
  `--sigmas qwen-image-2.1-turbo` schedule: 8 Euler steps at CFG 1.0, which also skips the
  negative-prompt pass. Without a published GGUF, convert the safetensors on a PC with
  `sd-cli -M convert -m <turbo>.safetensors -o qwen_image_2.1_turbo-Q4_K.gguf --type q4_K`.
- Check the model license before distributing a release that offers this preset.

## Other GGUF text models

ProsperoAI can also download and discover standalone `.gguf` text models. The pinned
[llama.cpp revision](https://github.com/ggml-org/llama.cpp/tree/cb7934c52ca8710994b2ecc19775ebefcfdb8d01)
includes loaders for the Llama, Qwen, Gemma, Phi, DeepSeek, Granite and other architectures,
plus chat-template handling. This means many GGUF models from those families can be tried
without adding model-specific app code. In Models, search for the model name or repository,
open its GGUF files, and download one file. The app will discover it in the installed-model
list. Examples include:

| Model example | GGUF repository | Starting quantization |
| --- | --- | --- |
| Llama 3.1 8B Instruct | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | Q4_K_M |
| Qwen3 8B | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | Q4_K_M |
| Gemma 3 4B IT | [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | Q4_K_M; text input only |
| Phi-4 Mini Instruct | [bartowski/microsoft_Phi-4-mini-instruct-GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF) | Q4_K_M |
| DeepSeek R1 Distill Qwen 7B | [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) | Q4_K_M |

These are compatibility candidates, not ProsperoAI-curated or PS5-verified presets. Start with a
single-file Q4 GGUF and a moderate context; the model's architecture, tokenizer and chat template
must be supported, its tensor types must have Vulkan kernels, and its weights plus context cache
must fit the console's available memory. Very large models, multimodal GGUFs, embedding models,
unsupported templates or quantizations can fail even when llama.cpp recognizes the architecture.
The list is representative rather than exhaustive; many fine-tunes of a supported family may
also work. See the pinned [architecture list](https://github.com/ggml-org/llama.cpp/blob/cb7934c52ca8710994b2ecc19775ebefcfdb8d01/src/llama-arch.cpp)
and [chat-template list](https://github.com/ggml-org/llama.cpp/blob/cb7934c52ca8710994b2ecc19775ebefcfdb8d01/src/llama-chat.cpp).

The curated presets were each exercised for three generation rounds on the test title; their
results are in [the retained 3× test record](performance/2026-10-14/pr27-all-models-3x-final.json).
That validation does not extend to the additional GGUF examples above.

Image generation currently writes both PNG and TGA: PNG is the shareable/downloadable output,
while the native conversation preview and archived image use TGA. Both formats are needed by
the current app flow.
