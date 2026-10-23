"""Paired signed synthetic Clip/QDQ rewrite, not dataset inference."""
import jetson_activation_boundary_probe as probe


if __name__ == '__main__':
    probe.SPEC_SCHEMA = 'rampq.gate05.code_clip_signed_probe/v1'
    probe.VARIANTS_TO_RUN = ('clip', 'code_clip')
    probe.COMPLETION_STATUS = 'SIGNED_CODE_CLIP_PAIRED_DIAGNOSTIC_SAVED'
    probe.main()
