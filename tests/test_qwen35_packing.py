from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.datasets.vlm.neat_packing_vlm import (
    PackedDatasetWrapper,
    _build_packed_vlm_sample,
    greedy_knapsack_vt_balanced,
)


class _Inner:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _sample(length: int, label_start: int = 100) -> dict:
    ids = torch.arange(length, dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": torch.arange(label_start, label_start + length, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "position_ids": torch.arange(length, dtype=torch.long),
    }


def test_packed_vlm_sample_preserves_boundaries_labels_and_reset_positions():
    packed = _build_packed_vlm_sample(
        [_sample(3, 10), _sample(2, 20)],
        pack_size=8,
        padding_idx=99,
    )

    assert packed["input_ids"].tolist() == [0, 1, 2, 0, 1]
    assert packed["labels"].tolist() == [10, 11, 12, 20, 21]
    assert packed["attention_mask"].tolist() == [1, 1, 1, 2, 2]
    assert packed["position_ids"].tolist() == [0, 1, 2, 0, 1]
    assert packed["seq_lens"] == [3, 2]
    assert packed["seq_lens_padded"] == [3, 2]


def test_packed_dataset_rejects_actual_sample_over_budget_by_default():
    inner = _Inner([_sample(7)])
    wrapper = PackedDatasetWrapper(
        inner_dataset=inner,
        bins=[[0]],
        pack_size=5,
        padding_idx=99,
        max_retries=1,
    )

    with pytest.raises(ValueError, match="exceeding pack_size"):
        wrapper[0]


def test_packed_dataset_rejects_pack_when_all_samples_are_dropped():
    inner = _Inner([_sample(7)])
    wrapper = PackedDatasetWrapper(
        inner_dataset=inner,
        bins=[[0]],
        pack_size=5,
        padding_idx=99,
        max_retries=1,
        drop_long_samples=True,
    )

    with pytest.raises(ValueError, match="produced no valid samples"):
        wrapper[0]


def test_vt_balanced_knapsack_keeps_all_indices_within_capacity():
    lengths = [4, 3, 2, 1]
    bins = greedy_knapsack_vt_balanced(lengths, max_length=5, visual_tokens=[8, 1, 7, 2])
    flattened = [idx for bucket in bins for idx in bucket]
    assert sorted(flattened) == [0, 1, 2, 3]
    assert all(sum(lengths[idx] for idx in bucket) <= 5 for bucket in bins)


def test_neat_pack_dataset_forwards_drop_policy_through_wrapper_config():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm

    packed = neat_pack_dataset_vlm(
        dataset=_Inner([_sample(3)]),
        pack_size=8,
        padding_idx=99,
        drop_long_samples=False,
        ds_raw=None,
    )

    assert isinstance(packed, PackedDatasetWrapper)
    assert packed.drop_long_samples is False


def _planner_processor():
    from types import SimpleNamespace

    return SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=14,
            merge_size=2,
            min_pixels=112 * 112,
            max_pixels=512 * 512,
            size={},
        )
    )


def test_smoke_dataset_has_varied_length_metadata_for_real_multi_sample_packing():
    from smoke_data import make_dataset

    samples = make_dataset("train")
    text_lengths = [sample["_text_tokens"] for sample in samples]
    image_meta = [sample["mm_inputs_meta"]["images_meta"] for sample in samples]

    assert len(set(text_lengths)) >= 4
    assert all(meta == [[112, 112]] for meta in image_meta)


def test_smoke_dataset_planner_creates_real_multi_sample_packs():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm
    from smoke_data import make_dataset

    raw = make_dataset("train")
    inner = _Inner([_sample(80 + (i % 4) * 20) for i in range(len(raw))])
    packed = neat_pack_dataset_vlm(
        dataset=inner,
        pack_size=512,
        padding_idx=0,
        ds_raw=raw,
        processor=_planner_processor(),
        balance_media_tokens=False,
        packing_ratio=0.9,
    )

    assert len(packed.bins) < len(raw)
    assert max(map(len, packed.bins)) > 1
    assert sorted(i for bucket in packed.bins for i in bucket) == list(range(len(raw)))


def test_real_multi_sample_pack_has_document_boundaries_and_positions():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm
    from smoke_data import make_dataset

    raw = make_dataset("train")[:4]
    inner = _Inner([_sample(40 + i * 10) for i in range(4)])
    packed = neat_pack_dataset_vlm(
        dataset=inner,
        pack_size=512,
        padding_idx=0,
        ds_raw=raw,
        processor=_planner_processor(),
        balance_media_tokens=False,
    )
    multi = next((idx for idx, bucket in enumerate(packed.bins) if len(bucket) > 1), None)
    assert multi is not None
    materialized = packed[multi]
    assert len(materialized["seq_lens"]) > 1
    assert materialized["attention_mask"].unique().tolist() == list(range(1, len(materialized["seq_lens"]) + 1))
    first_len = materialized["seq_lens"][0]
    assert materialized["position_ids"].tolist()[:first_len] == list(range(first_len))
    assert materialized["n_images"] == 0


def test_packed_vlm_sample_preserves_variable_resolution_tensor_media_as_list():
    samples = [
        {
            **_sample(3),
            "pixel_values": torch.zeros(1, 2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        },
        {
            **_sample(2),
            "pixel_values": torch.zeros(1, 4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    ]

    packed = _build_packed_vlm_sample(samples, pack_size=8, padding_idx=99)

    assert isinstance(packed["pixel_values"], list)
    assert [tuple(value.shape) for value in packed["pixel_values"]] == [(2, 3), (4, 3)]
    assert packed["image_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 2]]


def test_packed_vlm_sample_preserves_variable_resolution_video_tensor_media_as_list():
    samples = [
        {
            **_sample(3),
            "pixel_values_videos": torch.zeros(1, 2, 3),
            "video_grid_thw": torch.tensor([[1, 1, 2]]),
        },
        {
            **_sample(2),
            "pixel_values_videos": torch.zeros(1, 4, 3),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    ]

    packed = _build_packed_vlm_sample(samples, pack_size=8, padding_idx=99)

    assert isinstance(packed["pixel_values_videos"], list)
    assert [tuple(value.shape) for value in packed["pixel_values_videos"]] == [(2, 3), (4, 3)]
    assert packed["video_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 2]]


def test_neat_packed_vlm_collater_rejects_pack_over_fixed_token_budget():
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    with pytest.raises(ValueError, match="exceeds configured max_length"):
        neat_packed_vlm_collater([_sample(5)], padding_idx=99, max_length=4, attn_implementation="flash_attention_2")


@pytest.mark.parametrize(
    "media_key,grid_key,count_key,batch_count_key",
    [
        ("pixel_values", "image_grid_thw", "n_images", "n_images_per_sample"),
        ("pixel_values_videos", "video_grid_thw", "n_videos", "n_videos_per_sample"),
    ],
)
def test_neat_packed_vlm_collater_merges_tensor_and_list_media_values(
    media_key, grid_key, count_key, batch_count_key
):
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    def _packed(media):
        return {
            **_sample(2),
            "mm_token_type_ids": torch.zeros(2, dtype=torch.long),
            media_key: media,
            grid_key: torch.tensor([[1, 1, 2]], dtype=torch.long),
            count_key: 1,
        }

    batch = neat_packed_vlm_collater(
        [
            _packed(torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            _packed([torch.full((4, 3), 9.0)]),
        ],
        attn_implementation="flash_attention_2",
    )

    assert isinstance(batch[media_key], list)
    assert [tuple(value.shape) for value in batch[media_key]] == [(3,), (3,), (4, 3)]
    assert [value.dtype for value in batch[media_key]] == [torch.bfloat16] * 3
    assert batch[media_key][0].tolist() == [0, 1, 2]
    assert batch[media_key][1].tolist() == [3, 4, 5]
    assert batch[media_key][2].tolist() == [[9, 9, 9]] * 4
    assert batch[batch_count_key].tolist() == [1, 1]


@pytest.mark.parametrize(
    "media_key,grid_key", [("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")]
)
def test_packed_builder_and_collater_keep_mixed_resolution_media_order(media_key, grid_key):
    """A fixed-resolution pack and a variable-resolution pack share one batch."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    def sample(patches, value):
        return {
            **_sample(2),
            media_key: torch.full((1, patches, 3), value),
            grid_key: torch.tensor([[1, 1, patches]]),
        }

    fixed = _build_packed_vlm_sample([sample(2, 1.0), sample(2, 2.0)], pack_size=8, padding_idx=99)
    variable = _build_packed_vlm_sample([sample(2, 3.0), sample(4, 4.0)], pack_size=8, padding_idx=99)
    assert isinstance(fixed[media_key], torch.Tensor)
    assert isinstance(variable[media_key], list)
    batch = neat_packed_vlm_collater([fixed, variable], max_length=8, attn_implementation="sdpa")
    assert [tuple(value.shape) for value in batch[media_key]] == [(2, 3), (2, 3), (2, 3), (4, 3)]
    assert [value[0, 0].item() for value in batch[media_key]] == [1, 2, 3, 4]
    assert batch[grid_key].tolist() == [[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 4]]
    assert batch["labels"].tolist() == [[100, 101, 100, 101, -100, -100, -100, -100]] * 2


@pytest.mark.parametrize(
    "media_key,grid_key", [("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")]
)
def test_qwen_patch_tensors_remain_concatenated_for_different_patch_counts(media_key, grid_key):
    """Qwen's [patches, features] path must not turn variable patch counts into lists."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    samples = [
        {**_sample(2), media_key: torch.full((2, 3), 1.0), grid_key: torch.tensor([[1, 1, 2]])},
        {**_sample(3), media_key: torch.full((4, 3), 2.0), grid_key: torch.tensor([[1, 2, 2]])},
    ]
    packed = _build_packed_vlm_sample(samples, pack_size=8, padding_idx=99)
    batch = neat_packed_vlm_collater([packed], max_length=8, attn_implementation="sdpa")
    assert isinstance(batch[media_key], torch.Tensor)
    assert tuple(batch[media_key].shape) == (6, 3)
    assert batch[media_key].dtype == torch.bfloat16
    assert batch[media_key].tolist() == [[1, 1, 1]] * 2 + [[2, 2, 2]] * 4


def _text_sample(length: int) -> dict:
    return {"input_ids": list(range(length)), "labels": list(range(length))}


def test_thd_packing_aligns_each_document_to_requested_multiple():
    from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset

    packed = pack_dataset(
        [_text_sample(3), _text_sample(2)],
        split=None,
        packed_sequence_size=8,
        padding_idx=99,
        pad_to_multiple_of=4,
    )

    row = packed[0]
    assert row["seq_lens"] == [3, 2]
    assert row["seq_lens_padded"] == [4, 4]
    assert len(row["input_ids"]) == 8
    assert row["labels"][-1] == -100


def test_thd_packing_rejects_budget_not_divisible_by_alignment_multiple():
    from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset

    with pytest.raises(ValueError, match="divisible by positive pad_to_multiple_of"):
        pack_dataset(
            [_text_sample(3)],
            split=None,
            packed_sequence_size=7,
            padding_idx=99,
            pad_to_multiple_of=4,
        )


def test_vlm_neat_config_forwards_explicit_sequence_alignment():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig

    config = NeatPackConfig(
        pack_size=16,
        packing_format="neat",
        sequence_alignment=4,
    )
    packed = config.build(
        dataset=_Inner([_sample(5)]),
        padding_idx=99,
        ds_raw=None,
        cp_size=1,
    )

    assert packed.sequence_alignment == 4
    materialized = packed[0]
    assert materialized["seq_lens"] == [4]
    assert materialized["seq_lens_padded"] == [4]


def test_vlm_neat_config_rejects_alignment_that_breaks_mrope():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig

    config = NeatPackConfig(pack_size=16, packing_format="neat", sequence_alignment=4)
    with pytest.raises(NotImplementedError, match="mRoPE"):
        config.build(
            dataset=_Inner([_sample(5)]),
            padding_idx=99,
            ds_raw=None,
            get_rope_index=lambda **kwargs: None,
            cp_size=1,
        )


def test_vlm_thd_explicit_alignment_combines_with_context_parallel_geometry():
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig

    config = NeatPackConfig(pack_size=16, packing_format="thd", sequence_alignment=2)
    packed = config.build(
        dataset=_Inner([_sample(5)]),
        padding_idx=99,
        ds_raw=None,
        cp_size=2,
    )

    assert packed.sequence_alignment == 4
