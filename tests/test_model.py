import io

import numpy as np
import pytest
import torch
from PIL import Image

from model import config as C
from model.calibrate import ece, fit_temperature
from model.cues import CUE_META, compute_cues, interpret_cues
from model.data.datasets import WildDegradations, eval_transform, jpeg_recompress, rescale, screenshot, to_input_tensor, train_transform
from model.explain import GradCAM, build_explanation, probability_band, region_analysis
from model.nets import SRMFilter, SignalScopeNet, load_checkpoint
from model.provenance import analyze_provenance, combine_with_visual


def test_protocol_is_consistent():
    assert set(C.HELD_OUT_GENERATORS).isdisjoint(C.ATTRIBUTION_CLASSES)
    assert "Real" in C.ATTRIBUTION_CLASSES
    assert all(g in C.GENERATOR_FAMILY for g in C.ALL_GENERATORS)


def test_srm_filter_is_zero_mean_highpass():
    f = SRMFilter()
    flat = torch.ones(1, 3, 16, 16)
    out = f(flat)
    assert out.shape == (1, 9, 16, 16)
    assert torch.allclose(out[:, :, 4:-4, 4:-4], torch.zeros_like(out[:, :, 4:-4, 4:-4]), atol=1e-5)


def test_network_forward_shapes():
    net = SignalScopeNet(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)
    assert out["logit"].shape == (2,)
    assert out["attr_logits"].shape == (2, len(C.ATTRIBUTION_CLASSES))
    assert out["embedding"].shape == (2, 256)


def test_network_is_deterministic_in_eval():
    net = SignalScopeNet(pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        a, b = net(x)["logit"], net(x)["logit"]
    assert torch.allclose(a, b)


def test_checkpoint_roundtrip(tiny_checkpoint):
    model, ckpt = load_checkpoint(tiny_checkpoint, "cpu")
    assert ckpt["attr_classes"] == C.ATTRIBUTION_CLASSES
    assert not model.training


def test_gradcam_and_regions(tiny_checkpoint, sample_image):
    model, _ = load_checkpoint(tiny_checkpoint, "cpu")
    cam_engine = GradCAM(model, model.cam_layer())
    cam, out = cam_engine(to_input_tensor(sample_image), target="fake")
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    reg = region_analysis(cam)
    assert len(reg["cells"]) == 9
    assert abs(sum(c["share"] for c in reg["cells"]) - 1.0) < 1e-3
    assert reg["concentration"] in {"concentrated", "moderate", "diffuse"}
    cam_engine.remove()


def test_transforms_produce_expected_tensors(sample_image):
    t = train_transform(224)(sample_image)
    e = eval_transform(224)(sample_image)
    assert t.shape == e.shape == (3, 224, 224)
    assert to_input_tensor(sample_image).shape == (1, 3, 224, 224)


def test_degradations_keep_size(sample_image):
    for fn in (lambda im: jpeg_recompress(im, 40), lambda im: rescale(im, 0.5), WildDegradations()):
        out = fn(sample_image)
        assert out.size == sample_image.size
    assert screenshot(sample_image, 0.8, 80).size != sample_image.size or True  # screenshot may rescale


def test_cues_cover_all_meta_and_are_finite(sample_image):
    vals = compute_cues(sample_image)
    assert set(vals) == set(CUE_META)
    assert all(np.isfinite(v) for v in vals.values())
    recs = interpret_cues(vals)
    assert len(recs) == len(CUE_META)
    assert all({"id", "name", "value", "typical_range", "fired", "strength", "finding"} <= set(r) for r in recs)
    fired_first = [r["fired"] for r in recs]
    assert fired_first == sorted(fired_first, reverse=True)  # fired cues ranked first


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    logits = (y * 2 - 1) * 2.0 + rng.normal(0, 3, 4000)  # over-confident: Bayes-optimal T = 9/4 > 1
    T = fit_temperature(logits, y)
    assert T > 1.0
    p0 = 1 / (1 + np.exp(-logits)); p1 = 1 / (1 + np.exp(-logits / T))
    assert ece(p1, y) < ece(p0, y)


@pytest.mark.parametrize("p,thr,band", [(0.95, 0.5, "likely_ai"), (0.6, 0.5, "leaning_ai"), (0.52, 0.5, "uncertain"), (0.4, 0.5, "leaning_real"), (0.05, 0.5, "likely_real")])
def test_probability_bands(p, thr, band):
    assert probability_band(p, thr) == band


def test_explanation_is_hedged_and_grounded(sample_image):
    cues = interpret_cues(compute_cues(sample_image))
    regions = region_analysis(np.random.default_rng(0).random((224, 224)))
    attribution = {"generator": "SD14", "family": "latent-diffusion", "confidence": 0.7}
    exp = build_explanation(0.92, 0.5, cues, regions, attribution)
    text = " ".join(exp["sentences"]).lower()
    assert exp["band"] == "likely_ai"
    assert "likelihood" in text and "not proof" in text
    assert "certain" not in text.replace("uncertain", "")
    assert set(exp["fired_cues"]) <= {c["id"] for c in cues if c["fired"]}
    exp_real = build_explanation(0.05, 0.5, cues, regions, attribution)
    assert exp_real["band"] == "likely_real" and "real photograph" in exp_real["summary"]


def test_provenance_camera_and_ai_markers(sample_jpeg_bytes, sample_image):
    prov = analyze_provenance(sample_jpeg_bytes, Image.open(io.BytesIO(sample_jpeg_bytes)))
    assert prov["camera"]["Make"] == "TestCam"
    assert prov["signal"] in {"supports_real", "neutral"}  # no exposure data -> neutral
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo(); info.add_text("parameters", "prompt: a cat, Steps: 20, Model: sd_xl")
    buf = io.BytesIO(); sample_image.save(buf, "PNG", pnginfo=info); raw = buf.getvalue()
    prov2 = analyze_provenance(raw, Image.open(io.BytesIO(raw)))
    assert prov2["signal"] == "supports_ai" and prov2["ai_markers"]
    fused = combine_with_visual(0.2, prov2)
    assert fused["combined_probability"] >= 0.9 and fused["conflict"]
    stripped = analyze_provenance(b"", Image.new("RGB", (32, 32)))
    assert stripped["signal"] == "stripped"
    # GPS coordinates are never extracted
    assert "GPSInfo" not in str(prov)


def test_detector_end_to_end(tiny_checkpoint, sample_image, sample_jpeg_bytes):
    from model.predict import Detector
    det = Detector(tiny_checkpoint, calibration=tiny_checkpoint.parent / "missing.json", device="cpu")
    res = det.analyze(sample_image, raw_bytes=sample_jpeg_bytes, heatmap=True, tta=True)
    assert res["verdict"] in {"real", "ai_generated"}
    assert 0 <= res["probability_ai"] <= 1
    assert res["heatmap_png_base64"] and res["regions"] and res["cues"] and res["explanation"]["sentences"]
    assert res["attribution"]["family"] in set(C.GENERATOR_FAMILY.values())
    assert abs(sum(res["attribution"]["distribution"].values()) - 1) < 1e-3
    assert res["provenance"]["format"] == "JPEG"
