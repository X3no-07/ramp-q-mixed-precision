"""Paired first-Conv Clip intervention; no complete-model repair claim."""
import jetson_firstconv_numeric_probe as probe


if __name__ == '__main__':
    probe.SPEC_SCHEMA = 'rampq.gate05.firstconv_clip_intervention/v1'
    probe.RESULT_SCHEMA = 'rampq.gate05.firstconv_clip_intervention_native/v1'
    probe.COMPLETION_STATUS = 'TWO_ISOLATED_FIRSTCONV_CLIP_INTERVENTIONS_SAVED'
    probe.CASES = (('clip_fp16', 'clip', True), ('no_clip_fp16', 'no_clip', True))
    probe.QUANTIZED_VARIANTS = {'clip', 'no_clip'}
    probe.CROSS_CASE_PAIRS = (('clip_fp16', 'no_clip_fp16'),)
    probe.main()
