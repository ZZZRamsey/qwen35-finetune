# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
import importlib.metadata
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple, Type, Union

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


# The public source tree only retains Qwen3.5 model implementations.  Keep the
# compatibility hook consumed by model_init.py, but do not register config
# classes for model families that were intentionally pruned from this release.
_CUSTOM_CONFIG_REGISTRATIONS: Dict[str, Tuple[str, str]] = {}


def resolve_custom_config_cls(model_type: str) -> Type[PretrainedConfig] | None:
    """Return a retained custom config class for ``model_type``, if any."""
    registration = _CUSTOM_CONFIG_REGISTRATIONS.get(model_type)
    if registration is None:
        return None
    module_path, class_name = registration
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        logger.debug("Custom config unavailable for model_type=%s", model_type, exc_info=True)
        return None

# Static mapping: architecture name → (module_path, class_name[, tags]).
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
# Optional third element is a set of tags (e.g. {"retrieval"}) used by
# downstream code to classify model archs without importing them.
MODEL_ARCH_MAPPING = OrderedDict(
    [
        ("Qwen3_5ForCausalLM", ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForCausalLM")),
        ("Qwen3_5ForConditionalGeneration", ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration")),
        ("Qwen3_5MoeForCausalLM", ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForCausalLM")),
        ("Qwen3_5MoeForConditionalGeneration", ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration")),
    ]
)

class _LazyArchMapping:
    """Lazy-loading mapping from architecture name to model class.

    Inspired by HuggingFace transformers' ``_LazyAutoMapping``.  Entries from the
    static ``auto_map`` are imported on first access and cached.  Additional entries
    can be added at runtime via ``register``.
    """

    def __init__(self, auto_map: Union[OrderedDict, Dict[str, tuple], None] = None):
        # Entries may be (module_path, class_name) or (module_path, class_name, tags).
        # Strip the optional tags and store them separately.
        self._auto_map: Dict[str, Tuple[str, str]] = OrderedDict()
        self._tags: Dict[str, set] = {}
        for key, value in (auto_map or {}).items():
            self._auto_map[key] = (value[0], value[1])
            if len(value) > 2:
                self._tags[key] = value[2]
        self._loaded: Dict[str, Type[nn.Module]] = {}
        self._extra: Dict[str, Type[nn.Module]] = {}
        self._modules: Dict[str, object] = {}

    def _load(self, key: str) -> Type[nn.Module]:
        if key in self._loaded:
            return self._loaded[key]
        module_path, class_name = self._auto_map[key]
        if module_path not in self._modules:
            self._modules[module_path] = importlib.import_module(module_path)
        cls = getattr(self._modules[module_path], class_name)
        self._loaded[key] = cls
        return cls

    def __contains__(self, key: str) -> bool:
        if key in self._extra or key in self._loaded:
            return True
        if key not in self._auto_map:
            return False
        try:
            self._load(key)
            return True
        except Exception:
            logger.debug("Model %s unavailable (import failed), removing from auto_map", key)
            self._auto_map.pop(key, None)
            return False

    def __getitem__(self, key: str) -> Type[nn.Module]:
        if key in self._extra:
            return self._extra[key]
        if key in self._auto_map:
            return self._load(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Type[nn.Module]) -> None:
        self._extra[key] = value

    def register(self, key: str, value: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a model class under the given architecture name.

        Conflicts, including built-in architectures, require ``exist_ok=True``.
        """
        if not exist_ok and (key in self._auto_map or key in self._extra):
            raise ValueError(f"Duplicated model implementation for {key}")

        self._auto_map.pop(key, None)
        self._loaded.pop(key, None)
        self._extra[key] = value

    def has_tag(self, key: str, tag: str) -> bool:
        """Return ``True`` if *key* was registered with *tag*."""
        return tag in self._tags.get(key, set())

    def keys_with_tag(self, tag: str) -> set:
        """Return all architecture names that have *tag*."""
        return {k for k, tags in self._tags.items() if tag in tags}

    def keys(self):
        return set(self._auto_map.keys()) | set(self._extra.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"_LazyArchMapping(auto_map={len(self._auto_map)}, extra={len(self._extra)}, loaded={len(self._loaded)})"


@dataclass
class _ModelRegistry:
    model_arch_name_to_cls: _LazyArchMapping = field(default=None)
    _retrieval_archs: set = field(default_factory=set)

    def __post_init__(self):
        if self.model_arch_name_to_cls is None:
            self.model_arch_name_to_cls = _LazyArchMapping(MODEL_ARCH_MAPPING)
        self._retrieval_archs = self.model_arch_name_to_cls.keys_with_tag("retrieval")
        self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Register architecture entry points without importing their modules."""
        mapping = self.model_arch_name_to_cls
        for ep in importlib.metadata.entry_points(group="nemo_automodel.architectures"):
            if ep.name in mapping.keys():
                logger.warning("Architecture %s is already registered; skipping entry point %s", ep.name, ep.value)
                continue
            module_path, _, class_name = ep.value.rpartition(":")
            if not module_path or not class_name:
                raise ValueError(f"Entry point {ep.name!r} value must be module.path:ClassName, got {ep.value!r}")
            mapping._auto_map[ep.name] = (module_path, class_name)

    @property
    def supported_models(self):
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def has_custom_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* has a custom (non-HF) implementation."""
        return arch_name in self.model_arch_name_to_cls

    def has_retrieval_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* is a registered retrieval/encoder architecture."""
        return arch_name in self._retrieval_archs

    def register_retrieval(self, arch_name: str) -> None:
        """Mark *arch_name* as a retrieval/encoder architecture."""
        self._retrieval_archs.add(arch_name)

    def resolve_custom_model_cls(self, architecture: str, config) -> Union[Type[nn.Module], None]:
        """Return the custom model class if it exists and supports *config*, else ``None``.

        Custom model classes may define a ``supports_config(config)`` classmethod
        to opt out for specific HF configs (e.g. a Mistral3 VLM with a dense
        Ministral3 text backbone instead of the expected Mistral4 MoE+MLA).
        """
        if architecture not in self.model_arch_name_to_cls:
            return None
        model_cls = self.model_arch_name_to_cls[architecture]
        if hasattr(model_cls, "supports_config") and not model_cls.supports_config(config):
            logger.info(
                "Custom model %s does not support config %s, falling back to HF",
                model_cls.__name__,
                type(config).__name__,
            )
            return None
        return model_cls

    def register(self, arch_name: str, model_cls: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a custom model class for a given architecture name."""
        self.model_arch_name_to_cls.register(arch_name, model_cls, exist_ok=exist_ok)


@lru_cache
def get_registry():
    return _ModelRegistry()


def register_architecture(arch_name: str, model_cls: type[nn.Module], *, exist_ok: bool = False) -> None:
    """Register a custom model class for an architecture name.

    Args:
        arch_name: Architecture name (e.g. ``"LlavaExampleNemotronForCausalLM"``).
        model_cls: The model class (not a string path; the class object itself).
        exist_ok: If True, replace an existing registration.

    Raises:
        ValueError: If *arch_name* is already a built-in or registered to a
            different class and *exist_ok* is False.
    """
    ModelRegistry.register(arch_name, model_cls, exist_ok=exist_ok)


ModelRegistry = get_registry()
