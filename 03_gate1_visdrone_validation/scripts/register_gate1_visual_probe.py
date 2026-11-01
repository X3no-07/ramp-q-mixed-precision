"""Record the finite information probe and start exactly one hidden supervisor."""
import json
from pathlib import Path
import subprocess
import sys
import gate1_visual_evidence_probe as V
P=V.P;ROOT=V.ROOT;OUT=V.OUT


def main():
    c=P.read(OUT/'contract.json');latest=ROOT/'artifacts/gate1/latest_execution_state.json';prior=P.read(latest)
    P.require(prior['snapshot']=='artifacts/gate1/current_execution_state_v44.json','Latest state moved')
    P.require(P.identity(ROOT/prior['snapshot'])['sha256']==prior['snapshot_sha256'],'Previous snapshot changed')
    P.require(not (OUT/'supervisor_launch_receipt.json').exists() and not (OUT/'attempt_01').exists(),'Existing visual information probe')
    report=ROOT/'docs/81_gate1_local_visual_information_feasibility_launch.md'
    text='# Gate1 Local Visual Information Feasibility Diagnosis Initiated\n\n2026-10-26.\n\nThe previous three-group/feature ablation study did not demonstrate stable superiority over the original scorer. This round verifies a new information question: retaining the original 22-dimensional score and bounding box features, and further incorporating local visual evidence from P3 derived from a frozen forward pass of the R module, whether it can provide additional true/false bounding box ranking information beyond the prefix fitting of the scorer.\n\nOnly one candidate is registered: a 64-dimensional center feature and a 64-dimensional center-minus-neighborhood-mean feature, concatenated with the original 22-dimensional features, resulting in a total of 150 dimensions plus a bias. The 8 surrounding sampling points are fixed at 0.75 times the width and height of the bounding box, sampled via bilinear interpolation; no ground truth, prefix, or source identity is provided. This is not a new method proposal, nor a full reproduction of IoU-Net.\n\nReuse the original 512 scorer-fitting images and 213716 existing soft-quality labels, with image-equal loss, fitting-image-only normalization, zero-residual initialization,100 CPU float64 Adam updates, lr0.03, and L2 coefficient 0.0005. The detector remains frozen throughout. No candidates, feature layouts, hyperparameters, or seeds are added.\n\nThe diagnosis uses only 178 original scorer-fitted prefix images that do not overlap with the prefix calibration set. These images were previously used for calibration and belong to the original detector training data; this is an information feasibility check, not an independent test. The official IoU0.5/cap500 curve under the same R@0.25 false positive budget and the per-image equally weighted soft quality BCE are compared. A diagnostic signal with continued value is only recorded if both the original 22-dimensional scorer is strictly improved in TP retention and the BCE is strictly reduced; curve thresholds are only descriptive and not frozen for new deployment thresholds.\n\nHistorical intermediate features were not saved, requiring 690 new image forwards solely to extract new P3 information. For every image, all 8400 raw candidate outputs must match historical R bitwise. Intermediate feature maps and sampled arrays are saved for independent CPU recomputation. A deployment implementation could share the same backbone computation, but that possibility is not treated as a measured speed benefit here.\n\nImages, labels, and results for the 414 development images are not read, no new complete AP is run, and neither Selection nor Holdout payloads are read. Execution begins after 6 preflight checks, with no automatic retry after failure. All negative results are retained. Positive results would still require mechanism controls, frozen development validation, and independent scientific evidence; they would not immediately establish a successful paper.\n\nLocalization quality and ranking have prior work: [IoU-Net](https://arxiv.org/abs/1807.11590), [GFLV2](https://arxiv.org/abs/2011.12885). This round only uses a local visual linear probe to assess feasibility and does not claim originality, real INT8 performance, or end-to-end deployment validity based on this alone.\n\nEvidence directory: `artifacts/gate1/20261026_local_visual_information_probe_v1`\n'
    with report.open('x',encoding='utf-8') as f:f.write(text)
    snap=ROOT/'artifacts/gate1/current_execution_state_v45.json'
    P.write(snap,{'schema':'rampq.gate1.execution_state/v45','status':'REGISTERED_ONE_LOCAL_VISUAL_INFORMATION_PROBE','at_utc':P.now(),
        'previous_snapshot':prior,'contract':P.identity(OUT/'contract.json'),'report':P.identity(report),
        'new_candidates':1,'cumulative_scorer_candidates':7,'fit_images':512,'calibration_images':178,'real_image_forward_budget':690,
        'new_full_AP_jobs':0,'training_val_payload_access':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'scientific_acceptance':False,'actual_INT8':False,'no_automatic_retry':True})
    P.FT.atomic_json(latest,{'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v45.json',
        'snapshot_sha256':P.identity(snap)['sha256'],'note':'One new150feature local visual information probe:512fit/178scorer-fit-prefix-disjoint calibration only.690newforwards to obtain previously unsaved P3,require all8400raw bitwise equal;no developer AP,no new deploy threshold,no scientific/INT8 acceptance,Holdout sealed.'})
    argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/launch_gate1_visual_evidence_probe.py')]
    with (OUT/'supervisor_stdout.txt').open('x',encoding='utf-8') as stdout,(OUT/'supervisor_stderr.txt').open('x',encoding='utf-8') as stderr:
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    P.write(OUT/'supervisor_launch_receipt.json',{'at_utc':P.now(),'pid':proc.pid,'argv':argv,'automatic_retry':False})
    print(json.dumps({'status':'LAUNCHED_ONE_LOCAL_VISUAL_INFORMATION_PROBE','supervisor_pid':proc.pid,'snapshot':'v45','contract':P.identity(OUT/'contract.json')},ensure_ascii=True))


if __name__=='__main__':main()
