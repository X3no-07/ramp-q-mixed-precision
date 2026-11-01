# Gate 0.5: Localization of Numerical Discrepancies in the Full Reference



2026-10-19, continuing from [full q_only_G1 candidate fix](28_gate05_fused_prefix_and_full_candidate_validation.md). This round localizes the reference discrepancy of full q_all on the original fixed `000020.jpg`: small floating-point differences in the convolution backend cross quantization rounding boundaries and amplify layer by layer; after unifying Conv and SiLU backends in memory, the 88-layer convolution outputs match the original ONNX Runtime byte-by-byte, with final box difference of 0 and score maximum difference of approximately 1.31e-7. This is a counterfactual diagnosis of the full model; the original PyTorch/ORT guardrail still fails, thus Gate 0.5 is not passed.

All work in this round was performed on the local CPU. No board-side tasks, dataset evaluation, re-calibration, or training were executed. Original images, weights, calibration, thresholds, and numerical guardrails are all preserved.

# # Original Failure Accurately Reproduced

The newly run PyTorch and ORT main outputs match the original `20261018_int8/input_v1/local_raw_outputs.npz` arrays byte-by-byte. ORT with graph optimization disabled, CPU FP32, four threads, input preprocessing preserved as original. Both tolerance components were re-verified, and score statistics were not skipped solely due to failure of the first assertion:

| Main output section | Original rtol / atol | Maximum absolute difference | Elements exceeding original tolerance |
|---|---|---:|---:|
| xywh | 0.005 / 0.5 | 96.5704651 | 14,204 / 33,600 |
| 10-class scores | 0.02 / 0.002 | 0.0729592 | 47 / 84,000 |

Therefore, this is not an issue arising from the irreproducibility of old logs. The original state remains `LOCAL_QDQ_PREFLIGHT_FAILED`, with no new tolerance, and without changing the criterion from matching prediction counts per image to numerical pass standards.

# # Same Convolution Input Comparison

From the original q_all, 88 independent Conv + original weight Q/DQ subgraphs are extracted, preserving the convolution properties and Q/DQ nodes along with their initializers in original byte form, while explicitly feeding the activation DQ outputs as inputs. First, the full ORT's own activation DQ inputs are replayed, and all 88 convolution outputs match the original full ORT on a byte-by-byte basis, indicating that the actual convolution results are not altered by the graph splitting process.

Then, PyTorch-recorded activation DQ inputs are fed in, and the convolution outputs are compared against the original PyTorch outputs under the same input: the maximum absolute difference across the 88 layers is `2.86102294921875e-5`. Relatively, when both sides run independently, the maximum difference in convolution outputs across the 88 layers is `3.26055645942688`.

This indicates that with synchronized inputs, local convolution differences are minimal, while independent runtime leads to accumulating divergence in intermediate values. This is a replay diagnosis, not a standalone validation of the complete model; thus, it cannot be used to declare the original numerical guardrail passed.

Independent weight-only Q/DQ image checks examined 2, 576, 192 weight values, with numerical differences of 0; however, a total of 28, 476 weights across 87 layers exhibit bit-level differences, all between PyTorch `-0` and ORT `+0`. Among these, 28, 473 originated from negative values rounded to zero, and 3 of the original weights were inherently negative zeros. The initial boundary report used `source_qdq_input_and_weight_bitexact` with a numerical assertion allowing positive and negative zeros to be equal, which was overly strong; the original report is retained, with supplementary review clearly correcting it to numerical consistency. Four groups of genuine PyTorch floating-point DQ and ORT DQ each show 459, 991 bit-level differences between positive and negative zeros, with the numerical difference remaining at 0; the overall digit-level conclusion remains unchanged.

# # First Quantization Boundary Cross-Check

When the first Conv uses identical inputs, quantized weights, and biases, the maximum absolute difference between PyTorch and ORT is `1.621246337890625e-5`. After passing through the first SiLU, at the input of the second Conv where quantization occurs, there are 3 out of 1,638,400 bit-level discrepancies, corresponding to a quantization step size of `0.7481081485748291`.

Diagnosis swaps the Conv, SiLU, and quantizer across two backends:

- The Conv source is fixed, only the SiLU is switched; the bit-level differences in the second quantizer remain unchanged.
Switch the Conv source to flip the entire bit pattern at exactly 3 positions.
- The four SiLU inputs are separately fed into both PyTorch and ORT quantizers, and both bit-level and DQ numerical outputs are consistent across the two.
- The bare ORT convolution is replayed with dynamic x/w/b inputs, matching the original prefix output of the first Conv, and the split context has been verified.

The current fixed-grid locations of 3 are `[0,7,105,184]`,`[0,7,176,138]`,`[0,7,177,144]`. After dividing the SiLU output by the quantization scale, it lies within 0.5 on either side, with a distance to the boundaries of approximately 0.5e-6 to 2.4e-6 integer digit units. The original value, floating-point ULP, normalized value, and the four cross-check results are all preserved.

This supports a well-defined causal conclusion: the earliest quantization code separation at 3 is caused by floating-point computation differences in the first convolution crossing rounding boundaries; this site is not due to differences in the quantizer rules nor to code changes induced by switching SiLU. It does not exclude the possibility of additional backend discrepancies in subsequent network layers, and cannot equate the three locations directly with the sole source of errors across the entire model.

# # End-of-Chain Operator Replay

All group convolution outputs from the original graph are sliced to match the recorded PyTorch outputs, and then the main output is reverse-cropped, requiring only three classification heads and four outputs from the DFL convolution as inputs. The remaining 36 nodes contain no convolutions, including operations such as reshape, slice, coordinate decoding, and sigmoid.

The terminal subgraph shows a difference of 0 relative to PyTorch's main output boxes, with a maximum score difference of `1.3146405919428616e-7`. This indicates that, under consistent recorded terminal inputs, the remaining decoding operations do not reproduce the original 96.57 box discrepancy. This subgraph has been decoupled from upstream dependencies and is solely used for localization; it cannot replace the original full model.

# # Complete Operator Backend Comparison

Within the in-memory PyTorch model, the original activation `round/clamp/multiply`, weights, convolution properties, and other PyTorch operations are preserved. The 88 convolutions are temporarily delegated to separate ORT sessions. First, the session context is validated using the original full ORT's activation DQ inputs.

First, single-threaded execution showed minor discrepancies at 34/88 layers (maximum difference 1.6212463e-5), so no full counterfactual was performed for this configuration. Failure control and code retention were not executed. Instead, a four-threaded run consistent with the original full execution was adopted, and a control revalidation was conducted in a new directory. The outputs from all 88 sessions matched the original full ORT and prior replay byte-by-byte. Only then were two full counterfactuals executed.

| Reference implementation, against original full ORT | Maximum absolute difference in box coordinates | Maximum absolute difference in score | Number of box/score violations under original tolerance |
|---|---:|---:|---:|
| Original PyTorch Q/DQ | 96.5704651 | 0.0729592 | 14,204 / 47 |
| PyTorch Q/DQ plus ORT Conv, all else in PyTorch | 0 | 5.6009958e-6 | 0 / 0 |
| Above implementation with unified ORT SiLU | 0 | 1.3061498e-7 | 0 / 0 |

Replacing Conv alone yields byte-identical outputs for 84 Conv layers; the other four layers are in the classification head. The first additional numerical divergence occurs at an activation-quantization input of `model.23.cv3.0.0.1.conv`, with a difference of 0.0744243; the subsequent three layers are also fully recorded. After SiLU is also replaced, all 88 Conv outputs are byte-identical and all 88 activation Q/DQ inputs are numerically equal, although signed zeros prevent claiming identical input bytes throughout. SiLU has 1 shared instance and 77 aliases, and is actually called 77 times in the second counterfactual; the shared instance was not omitted.

This forms a complete chain of localization: the original failure is reproducible; local input convolution errors are small; the earliest three quantization code changes are determined by Conv origins; after unifying Conv across the full run, convergence is significantly improved; and after unifying SiLU, all quantized convolution outputs become identical. This supports that the primary divergence on the current fixed graph stems from backend floating-point arithmetic and quantization boundary amplification, rather than inconsistencies in the current verified quantization parameters, weights, or quantizer rules. This conclusion cannot be extrapolated to all images, all operators, TensorRT engines, or independent datasets.

Both counterfactuals modified the reference implementation itself. Although they remain within the original tolerance, they cannot be used to upgrade the original PyTorch/ORT records to PASS, nor can they be used to claim the new implementation as a substitute for the original Gate 0 simulation reference. The model parameters, source images, and script files maintain consistent SHA values before and after, and the forward pass in memory is fully restored upon completion.

# # Evidence Locations and Next Steps

- `../artifacts/gate05/20261019_reference_parity_diagnostic/full_replay`: Full failure reproduction, 88-layer synchronized convolutions, and end-to-end replay.
- `../artifacts/gate05/20261019_reference_parity_diagnostic/first_boundary`: First input crossing the quantization boundary, original arrays, and code; the first attempt with incorrect input type is preserved separately.
- `../artifacts/gate05/20261019_reference_parity_diagnostic/peer_review`: Independent validation and sign-zero verification.
- `../artifacts/gate05/20261019_reference_parity_diagnostic/hybrid_reference`: Single-threaded failure control, four-threaded success control, two full operator backend counterfactuals, and independent array validation.

| Key Derived Report | SHA256 |
|---|---|
| full_replay/analysis.json | `bc16f28872eb2c1d52e3f42ed9c0852d41c38c5df84b9dfb091bc66433204942` |
| first_boundary/analysis.json (original fields retained, corrections appended below) | `f4ec401ce4ff9225fc1b4183a6a6532308e66105632f8143deee640a4a7d2c1d` |
| peer_review/root_replay_and_weight_review.json | `954424d585110e85394692e8c09a7b36f46dc0be1af20eabc29730ee70817342` |
| hybrid_reference/attempt_2_four_threads/analysis.json | `e30548ea4f618cf36cd154a0d804e8244ac7129b175feee898c488076c3efdb6` |
| hybrid_reference/completion_verification.json | `df9fff7c4452118cb1c75a0f04e4dab995e831f84b276bf8993ed62b04af1d30` |

Original arrays, diagnostic code, source file hashes before and after, two failed attempts, and manifest are all preserved; the main phase has concluded normally with no background board-side tasks. Independent review verified the preserved arrays, 88 control runs, full bit-level coordinates, original tolerance counts across branches, and current file hashes.

Complete numerical migration has not passed; Gate 0.5 is `INCOMPLETE`. This round has progressed from 'numerical root cause not identified' to a reproducible backend arithmetic explanation. The next step requires establishing a clear reference protocol that distinguishes semantic equivalence from backend numerical consistency, and documenting the relationship with the original Gate 0 reference. Subsequent review of fixes, generalization, and dataset acceptance will follow. No direct launch of the remaining engine matrices is initiated at this stage; tolerances are not relaxed, and failure elimination is not achieved by merely comparing detection counts alone.
