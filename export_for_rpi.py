"""
Exports the trained SkinSense model to a Raspberry Pi-ready INT8 ONNX file.

WHY ONNX INSTEAD OF PYTORCH'S OWN QUANTIZATION:
Your training log showed `torch.backends.quantized.supported_engines` was
only `['onednn']` on this Windows machine - not `qnnpack`, which is what
actual ARM/Raspberry Pi deployment needs. PyTorch's quantized weights are
packed specifically for whichever backend converted them (fbgemm/onednn vs
qnnpack use different memory layouts), so a model quantized here with
onednn will NOT run correctly on a Pi even if you copy the file over - and
qnnpack isn't available to convert with locally on this Windows build in
the first place.

Rather than fight that, this script exports a plain ONNX graph and lets ONNX
Runtime - which ships prebuilt wheels for Raspberry Pi OS (aarch64) and has
its own ARM/NEON-optimized INT8 kernels - do the quantization instead. No
Linux/WSL2/qnnpack wrangling needed; this runs fine right here on Windows.

USAGE (on your desktop, after readjusted_train.py has produced a best_model.pth):
    pip install onnx onnxruntime
    python export_for_rpi.py

OUTPUT:
    outputs/rpi_export/<model_name>_float.onnx      - full-precision reference
    outputs/rpi_export/<model_name>_int8_rpi.onnx   - what you copy to the Pi
    outputs/rpi_export/label_encoder.pkl             - copied alongside for convenience
"""
import os
import sys
import shutil

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readjusted_train import build_model, prepare_model_for_qat, TrainingConfig
from preprocessing_optimized import OUTPUT_DIR, IMAGE_SIZE


def export():
    config = TrainingConfig()
    device = torch.device('cpu')

    print(f"Rebuilding {config.MODEL_NAME} architecture (must match training exactly "
          f"for the checkpoint to load)...")
    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        pretrained=False,   # we're about to load OUR trained weights, not ImageNet's
        quantizable=config.USE_QAT
    )
    if config.USE_QAT:
        # Reconstructing with prepare_model_for_qat using the SAME backend as
        # training (not switching to qnnpack here) keeps the state_dict keys
        # identical, so the load below can use strict=True safely.
        model = prepare_model_for_qat(model, device, backend=config.QAT_BACKEND)

    best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Couldn't find {best_model_path} - run readjusted_train.py first."
        )
    state_dict = torch.load(best_model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded weights from: {best_model_path}")

    model.eval()

    if config.USE_QAT:
        # We want the QAT-ADAPTED FLOAT weights (the whole point of QAT is
        # that training nudges weights to be quantization-friendly), not the
        # fake-quant simulation ops themselves - ONNX Runtime will do the
        # real INT8 conversion next, on its own terms.
        model.apply(torch.ao.quantization.disable_fake_quant)
        model.apply(torch.ao.quantization.disable_observer)
        print("Disabled fake-quant simulation - exporting the underlying "
              "QAT-adapted float weights.")

        # Disabling isn't enough on its own: it only flips internal flags,
        # it doesn't remove the FakeQuantize modules from the graph. Every
        # one of them still gets traced as an `aten::fused_moving_avg_obs_
        # fake_quant` op - which the ONNX exporter has no symbolic mapping
        # for at all (not an opset issue - it's just unimplemented), and
        # which the dynamo tracer also chokes on separately (its unconditional
        # `if self.observer_enabled[0] == 1:` check triggers a data-dependent
        # guard it can't resolve). With both flags disabled, FakeQuantize(x)
        # is mathematically just x - so swapping every instance for
        # nn.Identity() removes the untraceable op with zero numerical
        # difference (the conv weights used are already the BN-fused
        # `scaled_weight` computed inside ConvBn2d/ConvBnReLU2d; the
        # FakeQuantize call after that was always a passthrough here).
        def strip_fake_quant(module):
            for name, child in module.named_children():
                if isinstance(child, torch.ao.quantization.FakeQuantizeBase):
                    setattr(module, name, torch.nn.Identity())
                else:
                    strip_fake_quant(child)

        strip_fake_quant(model)
        print("Stripped FakeQuantize modules (replaced with Identity) so "
              "the graph traces cleanly for ONNX export.")

    export_dir = os.path.join(OUTPUT_DIR, 'rpi_export')
    os.makedirs(export_dir, exist_ok=True)
    float_onnx_path = os.path.join(export_dir, f'{config.MODEL_NAME}_float.onnx')
    int8_onnx_path = os.path.join(export_dir, f'{config.MODEL_NAME}_int8_rpi.onnx')

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\nExporting to ONNX: {float_onnx_path}")
    # dynamo=False: force the legacy TorchScript-based exporter. The QAT-prepared
    # model still has FakeQuantize modules attached as forward hooks (SE-block
    # skip_mul, weight_fake_quant on convs, etc.) even with fake-quant/observer
    # disabled - disabling only flips internal flags, it doesn't remove the
    # hooks. FakeQuantize.forward() unconditionally starts with
    # `if self.observer_enabled[0] == 1:`, a plain tensor-valued Python `if`.
    # The new dynamo/torch.export-based exporter (torch.onnx.export's default
    # since recent PyTorch) tries to resolve that as a symbolic guard and
    # throws GuardOnDataDependentSymNode. The legacy tracer just evaluates it
    # as a concrete bool at trace time like it always did, so this avoids the
    # crash without touching the QAT plumbing.
    torch.onnx.export(
        model, dummy_input, float_onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17,
        dynamo=False
    )

    print("Applying INT8 quantization via ONNX Runtime (weights -> int8)...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime isn't installed. Run: pip install onnx onnxruntime"
        )
    quantize_dynamic(
        model_input=float_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8
    )

    # Copy the label encoder alongside so the Pi-side script has class names
    # without needing the rest of this project.
    label_encoder_src = os.path.join(OUTPUT_DIR, 'splits', 'label_encoder.pkl')
    if os.path.exists(label_encoder_src):
        shutil.copy(label_encoder_src, os.path.join(export_dir, 'label_encoder.pkl'))

    float_size = os.path.getsize(float_onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f"\nDone.")
    print(f"  Float ONNX: {float_onnx_path} ({float_size:.1f} MB)")
    print(f"  INT8 ONNX:  {int8_onnx_path} ({int8_size:.1f} MB)")
    print(f"\nCopy the whole '{export_dir}' folder to your Raspberry Pi and use "
          f"'{os.path.basename(int8_onnx_path)}' with onnxruntime.InferenceSession(...) "
          f"- see rpi_inference.py for a ready-made example.")

    print(f"\nNote: dynamic quantization only compresses weights - activations still "
          f"compute in float. It's simple and always safe, but for a bigger speed "
          f"win on this conv-heavy model, static quantization (calibrated on a "
          f"handful of real validation images) usually does better. Ask if you want "
          f"that version too.")


if __name__ == '__main__':
    export()
