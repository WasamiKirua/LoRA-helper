from copy import deepcopy

CONFIG = {
    "ui_hints": {
        "checkpoint_order": [
            "SDXL",
            "Flux",
            "Qwen-Image",
            "WAN-2.2",
            "SD-1.5",
            "Chroma",
            "Z-Image",
            "Qwen-Image-Edit",
            "custom",
        ],
        "use_case_order": ["identity", "likeness", "cinematic", "clothing", "pose", "abstract"],
    },
    "global_defaults": {
        "lora": {"filePrefix": "lora_img", "keyword": "my_subject"},
        "captionControl": {
            "guidance": "A detailed, high-quality description of the subject, its features, and the surrounding environment.",
            "guidanceStrength": 0.7,
            "negativeHints": "blurry, low quality, watermark, signature, text",
            "captionLength": "medium",
            "negativePreset": "auto",
            "strictFocus": False,
            "addQualityTags": True,
            "shuffleTopics": False,
            "subjectToken": "<subject>",
            "processingDelay": 1,
        },
        "promptTopics": ["photorealistic", "sharp focus", "4k", "", "", ""],
    },
    "checkpoints": {
        "SDXL": {
            "overrides": {"captionControl": {"addQualityTags": True}},
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "A photorealistic portrait of a specific person, focusing on consistent facial identity, sharp features, and neutral lighting. Capture the essence of the person for strong likeness.",
                        "strictFocus": True,
                        "guidanceStrength": 0.8,
                    },
                    "promptTopics": [
                        "photorealistic",
                        "skin texture",
                        "sharp focus",
                        "detailed eyes",
                        "neutral background",
                        "",
                    ],
                }
            },
        },
        "Flux": {
            "overrides": {"captionControl": {"captionLength": "short", "guidanceStrength": 0.6}},
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "A clear photo of a person. Focus on their main facial features and hair style.",
                        "strictFocus": True,
                    },
                    "promptTopics": ["natural photo", "clear face", "soft lighting", "", "", ""],
                }
            },
        },
        "Qwen-Image": {
            "overrides": {
                "captionControl": {
                    "addQualityTags": True,
                    "negativeHints": "watermark, jpeg artifacts, low quality",
                }
            },
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "[trigger], portrait, neutral background, capturing clear facial features.",
                        "captionLength": "short",
                        "strictFocus": True,
                    },
                    "promptTopics": ["Ultra HD", "4K", "sharp focus", "neutral background", "", ""],
                }
            },
        },
        "WAN-2.2": {
            "overrides": {
                "captionControl": {
                    "negativeHints": "watermark, blown highlights, motion blur",
                    "guidanceStrength": 0.8,
                }
            },
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "A video frame of a person. Describe their action, expression, and the shot type (e.g., medium shot, close-up).",
                        "captionLength": "medium",
                    },
                    "promptTopics": ["video frame", "4k", "consistent action", "clear motion", "", ""],
                }
            },
        },
        "SD-1.5": {
            "overrides": {
                "captionControl": {
                    "addQualityTags": False,
                    "negativePreset": "custom_merge",
                    "negativeHints": "extra fingers, deformed, ugly, bad anatomy",
                }
            },
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "portrait photo of a person, detailed face, clear features, (style)",
                        "strictFocus": True,
                        "captionLength": "short",
                    },
                    "promptTopics": ["detailed face", "sharp eyes", "soft lighting", "", "", ""],
                }
            },
        },
        "Z-Image": {
            "overrides": {
                "captionControl": {
                    "guidanceStrength": 0.6,
                    "captionLength": "medium",
                    "addQualityTags": True,
                    "negativeHints": "gibberish text, watermark",
                }
            },
            "use_cases": {
                "identity": {
                    "captionControl": {
                        "guidance": "Begin with the subject and deliver 1-2 natural sentences that describe the face, clothing, and framing. Include camera distance and lighting. Avoid bullet lists.",
                        "strictFocus": True,
                    },
                    "promptTopics": [
                        "photorealistic",
                        "soft diffused light",
                        "camera framing stated early",
                        "sharp focus",
                        "balanced composition",
                        "no gibberish text",
                    ],
                }
            },
        },
        "Chroma": {"use_cases": {}},
        "Qwen-Image-Edit": {"use_cases": {}},
        "custom": {"use_cases": {}},
    },
}

TOOLTIPS = {
    "backend": "Choose your preferred AI provider for generating captions.",
    "backendKey": "Your API key is stored locally in your browser and only sent to the selected provider.",
    "grokModel": "Pick the Grok vision model. grok-2-vision-latest works for images and NSFW cases.",
    "allowNsfw": "Grok will describe NSFW content honestly without censoring when enabled.",
    "trainingGoal": "Select the primary objective for your LoRA. This choice automatically applies best-practice settings for guidance, strength, and other controls.",
    "filePrefix": "A consistent name used for all your image and text files (e.g., 'audrey_0001.jpg').",
    "keyword": "The main trigger word to identify your subject. Use a unique, single word if possible.",
    "targetBaseModel": "Choose the base model you intend to use your LoRA with. Settings will be optimized for this choice.",
    "captionGuidance": "Instruct the AI on the desired style and content of the captions. Be descriptive. This is the most impactful setting.",
    "guidanceStrength": "Controls how strongly the AI adheres to your Caption Guidance. Higher values mean stricter adherence.",
    "negativeHints": "Tell the AI what to specifically AVOID describing in the captions (e.g., objects, colors, or styles).",
    "captionLength": "Sets the target length for the generated captions. 'Short' is often better for identity, 'Long' for style or cinematic LoRAs.",
    "negativePreset": "Automatically adds common negative prompts depending on the base model to improve quality.",
    "strictFocus": "Forces the AI to only describe the main subject, ignoring background and other elements. Ideal for character LoRAs.",
    "addQualityTags": "Appends common quality-enhancing tags like 'best quality', 'high resolution' to your captions.",
    "shuffleTopics": "Randomizes the order of Prompt Topics for each image, which can help prevent the model from associating topics with each other.",
    "subjectToken": "An additional token (e.g., <sks>) that can be used with your keyword for more stable subject identity.",
    "promptTopics": "Specific keywords or phrases that will be added to every caption to reinforce certain concepts like a style, an object, or a consistent color.",
    "processingDelay": "The delay in seconds between processing each image. Increasing this can help avoid API rate-limiting errors.",
}


def merge_deep(target: dict, *sources: dict) -> dict:
    for source in sources:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge_deep(target[key], value)
            else:
                target[key] = deepcopy(value)
    return target


def get_effective_config(checkpoint: str, use_case: str) -> dict:
    global_defaults = deepcopy(CONFIG["global_defaults"])
    checkpoint_config = CONFIG["checkpoints"].get(checkpoint, {})
    checkpoint_overrides = deepcopy(checkpoint_config.get("overrides", {}))
    use_case_settings = deepcopy(checkpoint_config.get("use_cases", {}).get(use_case, {}))
    return merge_deep(global_defaults, checkpoint_overrides, use_case_settings)
