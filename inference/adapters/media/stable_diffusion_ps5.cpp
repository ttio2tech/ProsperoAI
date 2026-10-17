#include "runtime_residency.hpp"
#include "ggml-vulkan.h"
static_assert(GGML_MAX_NAME == 160, "SD and llama must share the tensor ABI");
#include "stable-diffusion.h"
#include "../../../include/generated_output_path.hpp"
#include "../../../include/image_model_files.hpp"
#include "../../../third_party/miniz/miniz_tdef.h"
#include "../../../third_party/miniz/miniz.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C"
{
    int sceKernelDebugOutText(int, const char *);
    std::uint64_t sceKernelGetProcessTime(void);
}

namespace
{

sd_ctx_t *context;
char context_root[192];
// Set after a FLUX.3 image fails with all weights resident; later contexts in this run then
// load the encoder from disk per prompt, as Qwen Image 2.1 always does.
bool flux3_encoder_on_disk;
void (*progress_callback)(const char *);
bool backend_registered;

bool model_index_has(const char *root, const char *pipeline)
{
    char path[512];
    std::snprintf(path, sizeof(path), "%s/model_index.json", root);
    std::FILE *file = std::fopen(path, "rb");
    if (!file)
        return false;
    char contents[8192]{};
    const std::size_t count = std::fread(contents, 1, sizeof(contents) - 1, file);
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    return !failed && count > 0 && std::strstr(contents, pipeline) != nullptr;
}

bool is_sd3_model(const char *root)
{
    return root && model_index_has(root, "StableDiffusion3Pipeline");
}

bool is_flux2_klein_model(const char *root)
{
    return root && model_index_has(root, "Flux2KleinPipeline");
}

bool is_sdxl_checkpoint(const char *root)
{
    if (!root || !*root)
        return false;
    char path[512];
    if (std::snprintf(path, sizeof(path), "%s/sd_xl_base_1.0.safetensors", root) >=
        static_cast<int>(sizeof(path)))
        return false;
    std::FILE *file = std::fopen(path, "rb");
    if (!file)
        return false;
    std::fclose(file);
    return true;
}

// Hand-copied Qwen Image 2.1 and FLUX.3 Image Klein folders: the router lists them by the same
// rules (include/image_model_files.hpp). The pinned stable-diffusion.cpp gets FLUX.3 from
// tools/patches/stable-diffusion-flux3-klein.patch.
using prospero::image_models::Family;

prospero::image_models::Files image_model_files(const char *root)
{
    return root && *root ? prospero::image_models::detect(root) : prospero::image_models::Files{};
}

// The fixed schedule upstream's sd-cli selects with --sigmas qwen-image-2.1-turbo; the sigma
// count sets the step count (8).
float qwen_image_21_turbo_sigmas[] = {1.0f,     0.978453f, 0.95418f,  0.926626f, 0.89508f,
                                      0.845148f, 0.704534f, 0.414568f, 0.0f};

bool is_realistic_vision_v51(const char *root)
{
    if (!root || !*root)
        return false;
    char path[512];
    if (std::snprintf(path, sizeof(path), "%s/Realistic_Vision_V5.1_fp16-no-ema.safetensors", root) >=
        static_cast<int>(sizeof(path)))
        return false;
    std::FILE *file = std::fopen(path, "rb");
    if (!file)
        return false;
    std::fclose(file);
    return true;
}

void log_line(const char *format, ...)
{
    char line[512];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    sceKernelDebugOutText(0, line);
}

void sd_log(sd_log_level_t level, const char *message, void *)
{
    if (level >= SD_LOG_WARN)
        log_line("[prosperoai:sd] %s", message ? message : "");
}

void sd_progress(int step, int steps, float, void *)
{
    if (!progress_callback)
        return;
    char status[96];
    std::snprintf(status, sizeof(status), "Generating image on the GPU  ·  %d/%d", step, steps);
    progress_callback(status);
}

bool write_ppm(const char *path, const sd_image_t &image)
{
    if (!image.data || image.width == 0 || image.height == 0 || image.channel < 3)
        return false;
    std::FILE *file = std::fopen(path, "wbx");
    if (!file)
        return false;
    std::fprintf(file, "P6\n%u %u\n255\n", image.width, image.height);
    const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
    bool ok = true;
    if (image.channel == 3)
    {
        ok = std::fwrite(image.data, 3, pixels, file) == pixels;
    }
    else
    {
        for (std::size_t pixel = 0; pixel < pixels && ok; ++pixel)
            ok = std::fwrite(image.data + pixel * image.channel, 1, 3, file) == 3;
    }
    return std::fclose(file) == 0 && ok;
}

bool write_tga(const char *path, const sd_image_t &image)
{
    if (!image.data || image.width == 0 || image.height == 0 || image.width > 65535 ||
        image.height > 65535 || image.channel < 3)
        return false;
    std::FILE *file = std::fopen(path, "wbx");
    if (!file)
        return false;
    unsigned char header[18]{};
    header[2] = 2;
    header[12] = static_cast<unsigned char>(image.width);
    header[13] = static_cast<unsigned char>(image.width >> 8);
    header[14] = static_cast<unsigned char>(image.height);
    header[15] = static_cast<unsigned char>(image.height >> 8);
    header[16] = 32;
    header[17] = 0x28;
    bool ok = std::fwrite(header, 1, sizeof(header), file) == sizeof(header);
    std::vector<unsigned char> row(static_cast<std::size_t>(image.width) * 4);
    for (unsigned y = 0; y < image.height && ok; ++y)
    {
        for (unsigned x = 0; x < image.width; ++x)
        {
            const unsigned char *source =
                image.data + (static_cast<std::size_t>(y) * image.width + x) * image.channel;
            row[x * 4 + 0] = source[2];
            row[x * 4 + 1] = source[1];
            row[x * 4 + 2] = source[0];
            row[x * 4 + 3] = 255;
        }
        ok = std::fwrite(row.data(), 4, image.width, file) == image.width;
    }
    return std::fclose(file) == 0 && ok;
}

bool write_png(const char *path, const sd_image_t &image)
{
    if (!image.data || image.width == 0 || image.height == 0 ||
        (image.channel != 3 && image.channel != 4))
        return false;
    std::size_t png_size = 0;
    void *png = tdefl_write_image_to_png_file_in_memory(
        image.data, static_cast<int>(image.width), static_cast<int>(image.height),
        static_cast<int>(image.channel), &png_size);
    if (!png || png_size == 0)
    {
        mz_free(png);
        return false;
    }
    std::FILE *file = std::fopen(path, "wbx");
    const bool complete = file && std::fwrite(png, 1, png_size, file) == png_size;
    const bool closed = !file || std::fclose(file) == 0;
    mz_free(png);
    return complete && closed;
}

void release_context()
{
    prospero_runtime::set_model_resident(false);
    if (context)
    {
        free_sd_ctx(context);
        context = nullptr;
        context_root[0] = '\0';
    }
    log_line("[prosperoai] sd_context_released=1\n");
}

bool ensure_context(const char *root)
{
    if (context && std::strcmp(context_root, root) == 0)
        return true;
    release_context();
    if (!backend_registered)
    {
        if (ggml_backend_vk_get_device_count() == 0)
            return false;
        ggml_backend_register(ggml_backend_vk_reg());
        backend_registered = true;
    }
    sd_set_log_callback(sd_log, nullptr);
    sd_set_progress_callback(sd_progress, nullptr);

    sd_ctx_params_t parameters;
    sd_ctx_params_init(&parameters);
    const bool flux2_klein = is_flux2_klein_model(root);
    char diffusion_model_path[512];
    char llm_path[512];
    char vae_path[512];
    char checkpoint_path[512];
    const bool sdxl_checkpoint = is_sdxl_checkpoint(root);
    const bool sdxl = sdxl_checkpoint || model_index_has(root, "StableDiffusionXLPipeline");
    const bool realistic_vision = is_realistic_vision_v51(root);
    if (sdxl_checkpoint &&
        std::snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/sd_xl_base_1.0.safetensors", root) >=
            static_cast<int>(sizeof(checkpoint_path)))
        return false;
    if (realistic_vision &&
        (std::snprintf(checkpoint_path, sizeof(checkpoint_path),
                       "%s/Realistic_Vision_V5.1_fp16-no-ema.safetensors", root) >=
             static_cast<int>(sizeof(checkpoint_path)) ||
         std::snprintf(vae_path, sizeof(vae_path), "%s/vae.safetensors", root) >=
             static_cast<int>(sizeof(vae_path))))
        return false;
    if (flux2_klein &&
        (std::snprintf(diffusion_model_path, sizeof(diffusion_model_path), "%s/diffusion_model.gguf", root) >=
             static_cast<int>(sizeof(diffusion_model_path)) ||
         std::snprintf(llm_path, sizeof(llm_path), "%s/qwen3-4b-q2-k.gguf", root) >=
             static_cast<int>(sizeof(llm_path)) ||
         std::snprintf(vae_path, sizeof(vae_path), "%s/vae.safetensors", root) >=
             static_cast<int>(sizeof(vae_path))))
        return false;
    const prospero::image_models::Files files = image_model_files(root);
    const bool qwen_image_21 = files.family == Family::QwenImage21;
    const bool flux3 = files.family == Family::Flux3Klein;
    if ((qwen_image_21 || flux3) &&
        (std::snprintf(diffusion_model_path, sizeof(diffusion_model_path), "%s/%s", root,
                       files.diffusion.c_str()) >= static_cast<int>(sizeof(diffusion_model_path)) ||
         std::snprintf(llm_path, sizeof(llm_path), "%s/%s", root, files.encoder.c_str()) >=
             static_cast<int>(sizeof(llm_path)) ||
         std::snprintf(vae_path, sizeof(vae_path), "%s/%s", root, files.vae.c_str()) >=
             static_cast<int>(sizeof(vae_path))))
        return false;
    if (qwen_image_21 || flux3)
        log_line("[prosperoai] %s: diffusion=%s encoder=%s vae=%s\n", prospero::image_models::family_name(files),
                 files.diffusion.c_str(), files.encoder.c_str(), files.vae.c_str());
    // Diffusion transformers conditioned by a Qwen3-VL encoder, loaded as three files.
    const bool llm_dit = qwen_image_21 || flux3;
    parameters.model_path = (flux2_klein || llm_dit) ? nullptr
                            : (sdxl_checkpoint || realistic_vision) ? checkpoint_path
                                                                  : root;
    if (flux2_klein)
    {
        parameters.diffusion_model_path = diffusion_model_path;
        parameters.llm_path = llm_path;
        parameters.vae_path = vae_path;
        parameters.vae_format = SD_VAE_FORMAT_FLUX2;
    }
    if (llm_dit)
    {
        parameters.diffusion_model_path = diffusion_model_path;
        parameters.llm_path = llm_path;
        parameters.vae_path = vae_path;
    }
    if (realistic_vision)
        parameters.vae_path = vae_path;
    parameters.n_threads = 1;
    // Qwen Image 2.1 and FLUX.3 keep their GGUF weight types; the engine picks their flow
    // prediction, schedule shift and VAE format.
    parameters.wtype = llm_dit ? SD_TYPE_COUNT : SD_TYPE_F16;
    parameters.rng_type = CUDA_RNG;
    parameters.prediction = llm_dit                                ? PREDICTION_COUNT
                            : (is_sd3_model(root) || flux2_klein) ? FLOW_PRED
                                                                   : EPS_PRED;
    parameters.enable_mmap = false;
    parameters.eager_load = false;
    parameters.diffusion_conv_direct = false;
    parameters.vae_conv_direct = false;
    parameters.diffusion_flash_attn = false;
    parameters.backend = sdxl            ? "CPU,diffusion=Vulkan0,vae=CPU"
                         : llm_dit       ? "CPU,diffusion=Vulkan0,te=Vulkan0,vae=Vulkan0"
                                         : "CPU,diffusion=Vulkan0,vae=Vulkan0";
    // The 8B encoder (~4.7 GiB) and the 7B diffusion model (~3.9 GiB) do not both fit with
    // working buffers. Disk residency loads the encoder from its file on demand and lets the
    // engine evict it before the diffusion weights load, so one stage is resident at a time.
    // FLUX.3 (~5 GB at Q8_0) and its 4B encoder (~2.5 GB) fit together, so all its weights stay
    // loaded across images; a kept context then re-encodes only a new prompt.
    parameters.params_backend = (qwen_image_21 || (flux3 && flux3_encoder_on_disk)) ? "Vulkan0,te=disk"
                                                                                      : "Vulkan0";
    context = new_sd_ctx(&parameters);
    prospero_runtime::set_model_resident(context != nullptr);
    if (!context || !sd_ctx_supports_image_generation(context))
    {
        release_context();
        return false;
    }
    std::snprintf(context_root, sizeof(context_root), "%s", root);
    return true;
}

} // namespace

extern "C" void ps5_sd_shutdown()
{
    release_context();
}

extern "C" int ps5_sd_generate(const char *root, const char *prompt, char *response,
                               std::size_t response_capacity, std::uint64_t *elapsed_microseconds,
                               void (*progress)(const char *))
{
    constexpr char oracle_output[] = "/download0/ProsperoAI/work/image/prosperoai-image.ppm";
    char display_output[128];
    char png_output[128];
    if (!root || !*root || !prompt || !*prompt || !response || response_capacity == 0)
        return 1;
    if (!prospero_generated_output_path(display_output, sizeof(display_output), "image",
                                        "prosperoai-image", "tga") ||
        !prospero_generated_output_path(png_output, sizeof(png_output), "image",
                                        "prosperoai-image", "png"))
        return 1;
    const bool sd3 = is_sd3_model(root);
    const bool sdxl = model_index_has(root, "StableDiffusionXLPipeline") || is_sdxl_checkpoint(root);
    const bool flux2_klein = is_flux2_klein_model(root);
    const prospero::image_models::Files files = image_model_files(root);
    const bool qwen_image_21 = files.family == Family::QwenImage21;
    const bool qwen_image_21_turbo = qwen_image_21 && files.variant;
    const bool flux3 = files.family == Family::Flux3Klein;
    const bool flux3_base = flux3 && files.variant;
    progress_callback = progress;
    if (progress_callback)
        progress_callback(flux3_base            ? "Loading FLUX.3 Image Klein Base"
                          : flux3               ? "Loading FLUX.3 Image Klein"
                          : qwen_image_21_turbo ? "Loading Qwen Image 2.1 Turbo"
                          : qwen_image_21  ? "Loading Qwen Image 2.1"
                          : flux2_klein ? "Loading FLUX.2 Klein 4B"
                          : sdxl       ? "Loading Stable Diffusion XL"
                          : sd3        ? "Loading Stable Diffusion 3"
                                            : "Loading SD-Turbo");
    std::remove(oracle_output);
    std::remove(display_output);
    const std::uint64_t start = sceKernelGetProcessTime();
    if (!ensure_context(root))
    {
        std::snprintf(response, response_capacity, "Could not load the image model.");
        return 2;
    }

    sd_img_gen_params_t parameters;
    sd_img_gen_params_init(&parameters);
    parameters.prompt = prompt;
    parameters.negative_prompt = "";
    parameters.width = (sd3 || flux2_klein || qwen_image_21 || flux3) ? 768 : 512;
    parameters.height = parameters.width;
    parameters.seed = 42;
    parameters.batch_count = 1;
    parameters.sample_params.sample_steps = flux2_klein ? 4
                                            : flux3_base ? 20
                                            : flux3 ? 16
                                            : qwen_image_21_turbo ? 8
                                            : qwen_image_21 ? 20
                                            : sdxl ? 10
                                            : sd3 ? 28
                                            : is_realistic_vision_v51(root) ? 20
                                                  : 1;
    parameters.sample_params.sample_method = (sdxl || sd3 || flux2_klein || qwen_image_21 || flux3 ||
                                              is_realistic_vision_v51(root))
                                                 ? EULER_SAMPLE_METHOD
                                                 : EULER_A_SAMPLE_METHOD;
    if (flux2_klein)
        parameters.sample_params.scheduler = FLUX2_SCHEDULER;
    if (qwen_image_21_turbo)
    {
        parameters.sample_params.custom_sigmas = qwen_image_21_turbo_sigmas;
        parameters.sample_params.custom_sigmas_count =
            static_cast<int>(sizeof(qwen_image_21_turbo_sigmas) / sizeof(qwen_image_21_turbo_sigmas[0]));
    }
    if (flux2_klein)
        parameters.sample_params.guidance.distilled_guidance = 1.0f;
    parameters.sample_params.guidance.txt_cfg = is_realistic_vision_v51(root) ? 7.0f
                                               : flux3_base ? 3.5f
                                               : flux3 ? 1.0f
                                               : qwen_image_21_turbo ? 1.0f
                                               : qwen_image_21 ? 6.0f
                                               : sdxl ? 5.0f : sd3 ? 4.5f : 1.0f;

    sd_image_t *images = nullptr;
    int image_count = 0;
    bool generated = generate_image(context, &parameters, &images, &image_count);
    if (!generated && flux3 && !flux3_encoder_on_disk)
    {
        // The pinned engine cannot demote resident weights when memory runs out; retry once
        // with the encoder loaded per prompt instead of beside the diffusion weights.
        log_line("[prosperoai] flux3_retry_encoder_on_disk=1\n");
        free_sd_images(images, image_count);
        images = nullptr;
        image_count = 0;
        flux3_encoder_on_disk = true;
        release_context();
        generated = ensure_context(root) && generate_image(context, &parameters, &images, &image_count);
    }
    const bool written = generated && image_count == 1 && images &&
                         write_ppm(oracle_output, images[0]) &&
                         write_tga(display_output, images[0]) &&
                         write_png(png_output, images[0]);
    free_sd_images(images, image_count);
    // The diffusion weights stay resident in a kept context and cannot be evicted, so the next
    // request's encoder would have to fit beside them. A fresh context per image keeps the
    // encoder-then-diffusion order; reloading costs seconds against minutes of sampling.
    if (qwen_image_21)
        release_context();
    const std::uint64_t elapsed = sceKernelGetProcessTime() - start;
    if (elapsed_microseconds)
        *elapsed_microseconds = elapsed;
    progress_callback = nullptr;
    if (!written)
    {
        std::remove(display_output);
        std::remove(png_output);
        std::snprintf(response, response_capacity, "Image generation failed. Check the system log.");
        return 3;
    }
    std::snprintf(response, response_capacity,
                  "Image generated locally.\nPNG: %s\nPreview: %s", png_output, display_output);
    sceKernelDebugOutText(0, "[prosperoai] sd_complete=1\n");
    return 0;
}
