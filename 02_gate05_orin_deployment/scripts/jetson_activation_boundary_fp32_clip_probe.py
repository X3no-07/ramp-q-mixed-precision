"""Request original Clip to execute in FP32 without changing the ONNX graph."""
import jetson_activation_boundary_probe as probe


if __name__ == '__main__':
    probe.VARIANTS_TO_RUN = ('clip',)
    probe.FORCE_CLIP_PRECISION = 'fp32'
    probe.COMPLETION_STATUS = 'ORIGINAL_CLIP_FP32_CONSTRAINT_DIAGNOSTIC_SAVED'
    probe.main()
