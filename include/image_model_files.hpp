// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// Folders of hand-copied image models that stable-diffusion.cpp loads as three files
// (diffusion GGUF, Qwen3-VL encoder GGUF, VAE). The router lists them, the image adapter
// loads them and text-model discovery skips them, so all three use these rules.

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace prospero::image_models
{

enum class Family
{
    None,
    QwenImage21,
    Flux3Klein,
};

struct Files
{
    Family family = Family::None;
    // Qwen Image 2.1 Turbo weights, or FLUX.3 Image Klein Base weights.
    bool variant = false;
    std::string diffusion, encoder, vae;
    // When family is None but the folder holds an image VAE: what is missing.
    std::string missing;
};

inline bool starts_with_nocase(const std::string &name, const char *prefix)
{
    const std::size_t length = std::strlen(prefix);
    if (name.size() < length)
        return false;
    for (std::size_t i = 0; i < length; ++i)
        if (std::tolower(static_cast<unsigned char>(name[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i])))
            return false;
    return true;
}

inline bool ends_with_nocase(const std::string &name, const char *suffix)
{
    const std::size_t length = std::strlen(suffix);
    return name.size() >= length &&
           starts_with_nocase(name.substr(name.size() - length), suffix);
}

// Nonempty regular files directly in the folder, sorted by name.
inline std::vector<std::string> folder_files(const std::string &folder)
{
    std::vector<std::string> names;
    DIR *directory = opendir(folder.c_str());
    if (!directory)
        return names;
    while (const dirent *entry = readdir(directory))
    {
        const std::string name = entry->d_name;
        if (name.empty() || name[0] == '.')
            continue;
        struct stat info
        {
        };
        if (stat((folder + "/" + name).c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
            info.st_size > 0)
            names.push_back(name);
    }
    closedir(directory);
    std::sort(names.begin(), names.end());
    return names;
}

// The first file (by name) whose name starts with one of the prefixes and ends with the
// suffix, skipping names that start with an excluded prefix or contain "mmproj".
inline std::string pick(const std::vector<std::string> &names, std::initializer_list<const char *> prefixes,
                        const char *suffix, std::initializer_list<const char *> excluded = {})
{
    for (const std::string &name : names)
    {
        if (!ends_with_nocase(name, suffix) || starts_with_nocase(name, "mmproj") ||
            name.find("mmproj") != std::string::npos)
            continue;
        bool skip = false;
        for (const char *prefix : excluded)
            skip = skip || starts_with_nocase(name, prefix);
        if (skip)
            continue;
        for (const char *prefix : prefixes)
            if (starts_with_nocase(name, prefix))
                return name;
    }
    return {};
}

inline std::string first_present(const std::vector<std::string> &names,
                                 std::initializer_list<const char *> candidates)
{
    for (const char *candidate : candidates)
        for (const std::string &name : names)
            if (name == candidate)
                return name;
    return {};
}

constexpr const char *kQwenImage21Vae = "qwen_image_2.1_vae_bf16.safetensors";

inline Files detect(const std::string &folder)
{
    Files files;
    const std::vector<std::string> names = folder_files(folder);
    const auto describe_missing = [&](const char *what) {
        files.missing += files.missing.empty() ? what : std::string(", ") + what;
    };

    if (!first_present(names, {kQwenImage21Vae}).empty())
    {
        const std::string turbo =
            pick(names, {"qwen_image_2.1_turbo", "qwen-image-2.1-turbo"}, ".gguf");
        const std::string base = pick(names, {"qwen_image_2.1", "qwen-image-2.1"}, ".gguf",
                                      {"qwen_image_2.1_turbo", "qwen-image-2.1-turbo"});
        const std::string encoder = pick(names, {"Qwen3VL-8B-Instruct", "Qwen3-VL-8B-Instruct"}, ".gguf");
        if (turbo.empty() && base.empty())
            describe_missing("Qwen Image 2.1 diffusion GGUF (qwen_image_2.1*.gguf)");
        if (encoder.empty())
            describe_missing("Qwen3-VL 8B encoder GGUF (Qwen3VL-8B-Instruct*.gguf)");
        if (files.missing.empty())
        {
            files.family = Family::QwenImage21;
            files.variant = !turbo.empty();
            files.diffusion = files.variant ? turbo : base;
            files.encoder = encoder;
            files.vae = kQwenImage21Vae;
        }
        return files;
    }

    const std::string vae = first_present(names, {"ae.safetensors", "flux2-vae.safetensors", "flux2_ae.safetensors"});
    if (!vae.empty())
    {
        const std::string base = pick(names, {"flux-3-image-klein-base", "flux_3_image_klein_base"}, ".gguf");
        const std::string distilled =
            pick(names, {"flux-3-image-klein", "flux_3_image_klein"}, ".gguf",
                 {"flux-3-image-klein-base", "flux_3_image_klein_base"});
        const std::string encoder = pick(names, {"Qwen3VL-4B-Instruct", "Qwen3-VL-4B-Instruct"}, ".gguf");
        if (distilled.empty() && base.empty())
            describe_missing("FLUX.3 Image Klein diffusion GGUF (flux-3-image-klein*.gguf)");
        if (encoder.empty())
            describe_missing("Qwen3-VL 4B encoder GGUF (Qwen3VL-4B-Instruct*.gguf)");
        if (files.missing.empty())
        {
            files.family = Family::Flux3Klein;
            files.variant = distilled.empty();
            files.diffusion = files.variant ? base : distilled;
            files.encoder = encoder;
            files.vae = vae;
        }
    }
    return files;
}

// A folder holding one of these VAEs is an image model, complete or not; its GGUFs are not
// text models.
inline bool is_image_folder(const std::string &folder)
{
    return !first_present(folder_files(folder), {kQwenImage21Vae, "ae.safetensors", "flux2-vae.safetensors",
                                                 "flux2_ae.safetensors"})
                .empty();
}

inline const char *family_name(const Files &files)
{
    switch (files.family)
    {
    case Family::QwenImage21:
        return files.variant ? "Qwen Image 2.1 Turbo" : "Qwen Image 2.1";
    case Family::Flux3Klein:
        return files.variant ? "FLUX.3 Image Klein Base" : "FLUX.3 Image Klein";
    default:
        return "";
    }
}

} // namespace prospero::image_models
