"""Synthetic image QA for lifecycle testing, not an accuracy benchmark."""
from PIL import Image


def make_dataset(split="train"):
    # Keep visual inputs constant while varying explicit text estimates. The
    # planner consumes these fields before tokenization; the processor still
    # materializes and trains on the real conversation below.
    colors = [
        ("red", (255, 0, 0), 24, "short"),
        ("green", (0, 255, 0), 48, "medium " * 4),
        ("blue", (0, 0, 255), 72, "long " * 8),
        ("white", (255, 255, 255), 96, "longer " * 12),
    ]
    repeats = 4 if split == "train" else 1
    samples = []
    for name, rgb, text_tokens, length_marker in colors:
        for _ in range(repeats):
            prompt = (
                "What is the color of this image? Answer with one word. "
                f"{length_marker}"
            )
            samples.append(
                {
                    "conversation": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": Image.new("RGB", (112, 112), rgb)},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": name}]},
                    ],
                    "_text_tokens": text_tokens,
                    "mm_inputs_meta": {"images_meta": [[112, 112]], "videos_meta": []},
                }
            )
    return samples
