from __future__ import annotations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CONTENT_TYPES = {
    "roll": "product_roll",
    "installed": "installed_car",
    "install": "installation_process",
    "施工": "installation_process",
    "detail": "material_closeup",
    "closeup": "material_closeup",
    "before": "comparison",
    "after": "comparison",
    "pack": "packaging",
    "package": "packaging",
    "packaging": "packaging",
    "box": "packaging",
    "boxes": "packaging",
    "composite": "packaging_composite",
    "collage": "packaging_composite",
    "infographic": "text_composite",
    "poster": "poster",
    "portrait": "person_portrait",
    "person": "person_portrait",
    "human": "person_portrait",
    "scene": "scene_effect",
}

FILM_TYPES = {
    "ppf": "ppf_clear",
    "paint-protection": "ppf_clear",
    "clear": "ppf_clear",
    "tint": "window_tint",
    "window": "window_tint",
    "privacy": "window_tint",
    "wrap": "color_wrap",
    "color": "color_wrap",
    "vinyl": "color_wrap",
    "headlight": "headlight_film",
    "tool": "tool",
}

COLOR_FAMILIES = {
    "black": "black",
    "grey": "grey",
    "gray": "grey",
    "silver": "silver",
    "white": "white",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "purple": "purple",
    "gold": "gold",
    "transparent": "transparent",
    "clear": "transparent",
}

FINISHES = {
    "gloss": "gloss",
    "matte": "matte",
    "satin": "satin",
    "metallic": "metallic",
    "chrome": "chrome",
    "pearl": "pearl",
    "carbon": "carbon_fiber",
    "chameleon": "chameleon",
    "smoke": "smoke",
    "transparent": "transparent",
    "clear": "transparent",
}


def infer_from_name(name: str, vocabulary: dict[str, str], default: str = "unknown") -> str:
    lowered = name.lower()
    for token, value in vocabulary.items():
        if token in lowered:
            return value
    return default
