"""Run equivalent first-Conv code Clip plus corrected native importer attrs."""
import sys
import jetson_firstconv_numeric_probe as probe


def command(_old, root, out, variant):
    return [sys.executable, '-u', str(root / 'jetson_build_code_clip_importer_probe.py'),
            '--model', str(root / variant['model_file']), '--expected-sha256', variant['model_sha256'],
            '--output', str(out / 'model.engine')]


if __name__ == '__main__':
    probe.SPEC_SCHEMA = 'rampq.gate05.firstconv_code_clip_probe/v1'
    probe.RESULT_SCHEMA = 'rampq.gate05.firstconv_code_clip_native/v1'
    probe.COMPLETION_STATUS = 'FIRSTCONV_CODE_CLIP_IMPORTER_CORRECTION_DIAGNOSTIC_SAVED'
    probe.CASES = (('code_clip_importer_corrected', 'code_clip', False),)
    probe.QUANTIZED_VARIANTS = {'code_clip'}
    probe.CROSS_CASE_PAIRS = ()
    probe.BUILD_COMMAND_TRANSFORM = command
    probe.main()
