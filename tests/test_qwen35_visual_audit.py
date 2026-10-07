import torch
from torch import nn

from nemo_automodel.components.training.smoke_audit import (
    VisionTrainingAudit,
)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.language_model = nn.Linear(2, 2, bias=False)


def test_vision_audit_reports_finite_gradient_and_parameter_update():
    model = TinyVLM()
    audit = VisionTrainingAudit([model])

    assert audit.parameter_names == ["visual.weight"]
    before = audit.snapshot()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model.visual(torch.ones(1, 2)).sum()
    loss.backward()

    gradient_report = audit.inspect_gradients()
    assert gradient_report.trainable_count == 1
    assert gradient_report.gradient_count == 1
    assert gradient_report.all_finite is True

    optimizer.step()
    update_report = audit.compare_update(before)
    assert update_report.updated is True
    assert update_report.max_delta > 0
    assert update_report.all_finite is True


def test_vision_audit_does_not_accept_frozen_visual_parameters():
    model = TinyVLM()
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)

    audit = VisionTrainingAudit([model])
    assert audit.parameter_names == []
    assert audit.inspect_gradients().trainable_count == 0


def test_vision_audit_detects_nonfinite_gradient():
    model = TinyVLM()
    audit = VisionTrainingAudit([model])
    model.visual.weight.grad = torch.full_like(model.visual.weight, float("nan"))
    report = audit.inspect_gradients()
    assert report.gradient_count == 1
    assert report.all_finite is False


def test_vision_audit_detects_missing_update():
    model = TinyVLM()
    audit = VisionTrainingAudit([model])
    report = audit.compare_update(audit.snapshot())
    assert report.updated is False


def test_vlm_recipe_enables_visual_audit_only_when_configured():
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    model = TinyVLM()
    recipe = FinetuneRecipeForVLM({"smoke_audit": {"enabled": True, "min_delta": 1e-8}})
    recipe.model_parts = [model]
    recipe._configure_vision_audit()

    assert recipe.vision_audit is not None
    assert recipe.vision_audit.min_delta == 1e-8


def test_vlm_recipe_keeps_visual_audit_disabled_by_default():
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    recipe = FinetuneRecipeForVLM({})
    recipe.model_parts = [TinyVLM()]
    recipe._configure_vision_audit()

    assert recipe.vision_audit is None


def test_vlm_recipe_audit_policy_rejects_missing_visual_gradient():
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    recipe = FinetuneRecipeForVLM(
        {"smoke_audit": {"enabled": True, "require_finite_gradient": True}}
    )
    recipe.model_parts = [TinyVLM()]
    recipe._configure_vision_audit()

    report = recipe.vision_audit.inspect_gradients()
    try:
        recipe._validate_vision_gradient_report(report)
    except RuntimeError as exc:
        assert "visual gradient" in str(exc)
    else:
        raise AssertionError("missing visual gradient must fail an enabled audit")


def test_vlm_recipe_audit_policy_rejects_missing_parameter_update():
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM
    from nemo_automodel.components.training.smoke_audit import VisionUpdateReport

    recipe = FinetuneRecipeForVLM(
        {"smoke_audit": {"enabled": True, "require_parameter_update": True}}
    )
    recipe.model_parts = [TinyVLM()]
    recipe._configure_vision_audit()

    try:
        recipe._validate_vision_update_report(
            VisionUpdateReport(updated=False, all_finite=True, max_delta=0.0)
        )
    except RuntimeError as exc:
        assert "did not update" in str(exc)
    else:
        raise AssertionError("missing visual parameter update must fail an enabled audit")


def test_vlm_recipe_rejects_zero_visual_gradient_even_if_weight_decay_updates():
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    model = TinyVLM()
    recipe = FinetuneRecipeForVLM(
        {"smoke_audit": {"enabled": True, "require_finite_gradient": True}}
    )
    recipe.model_parts = [model]
    recipe._configure_vision_audit()
    before = recipe.vision_audit.snapshot()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    (model.visual(torch.ones(1, 2)).sum() * 0.0).backward()

    gradient_report = recipe.vision_audit.inspect_gradients()
    optimizer.step()
    update_report = recipe.vision_audit.compare_update(before)
    assert gradient_report.gradient_count == 1
    assert gradient_report.grad_norm == 0.0
    assert update_report.updated is True
    try:
        recipe._validate_vision_gradient_report(gradient_report)
    except RuntimeError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("zero visual gradient must fail an enabled audit")
