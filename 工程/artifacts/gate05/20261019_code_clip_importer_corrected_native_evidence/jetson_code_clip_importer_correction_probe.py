"""Execute valid code-domain Clip with explicitly corrected importer attrs."""
import sys
import jetson_activation_boundary_probe as probe


def command(_old, root, out, variant):
    return [sys.executable, '-u', str(root / 'jetson_build_code_clip_importer_probe.py'),
            '--model', str(root / variant['model_file']), '--expected-sha256', variant['model_sha256'],
            '--output', str(out / 'model.engine')]


if __name__ == '__main__':
    probe.SPEC_SCHEMA = 'rampq.gate05.code_clip_signed_probe/v1'
    probe.VARIANTS_TO_RUN = ('code_clip',)
    probe.BUILD_COMMAND_TRANSFORM = command
    probe.COMPLETION_STATUS = 'SIGNED_CODE_CLIP_IMPORTER_CORRECTION_DIAGNOSTIC_SAVED'
    probe.main()
