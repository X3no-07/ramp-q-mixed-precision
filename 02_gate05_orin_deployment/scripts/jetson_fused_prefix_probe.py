"""Execute paired fused first-Conv prefixes with one final DQ output only."""
import sys
import json
import jetson_activation_boundary_probe as probe


def command(_old, root, out, variant):
    name = next(name for name, value in json.loads((root / 'spec.json').read_text())['variants'].items()
                if value['model_sha256'] == variant['model_sha256'])
    return [sys.executable, '-u', str(root / 'jetson_build_fused_prefix_probe.py'),
            '--root', str(root), '--variant', name, '--output', str(out / 'model.engine')]


if __name__ == '__main__':
    probe.SPEC_SCHEMA = 'rampq.gate05.fused_prefix_probe/v1'
    probe.VARIANTS_TO_RUN = ('original', 'code_clip')
    probe.BUILD_COMMAND_TRANSFORM = command
    probe.MAX_BUILD_SECONDS = 300
    probe.CROSS_NATIVE_PAIRS = (('original', 'code_clip'),)
    probe.RESULT_STATUS = 'ISOLATED_FUSED_PREFIX_OUTPUTS_SAVED'
    probe.SCOPE_LIMITS = ('One original fixed image, two activation boundaries and one Conv+SiLU; '
                          'code_clip rewrites boundary semantics and corrects parsed attributes. '
                          'Actual SWISH/INT8 adoption remains inspector-dependent; no complete model pass.')
    probe.COMPLETION_STATUS = 'PAIRED_FUSED_PREFIX_OUTPUTS_SAVED'
    probe.main()
