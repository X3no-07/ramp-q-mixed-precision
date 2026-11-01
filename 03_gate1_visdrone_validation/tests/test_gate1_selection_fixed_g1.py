"""Pure/synthetic safeguards. Never opens real images/labels or runs a model."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gate1_selection_fixed_g1 as R


def authorization():
    return {"schema":"rampq.gate1.Selection_fixed_G1_human_access_decision/v1",
        "decision":"AUTHORIZED_REPORT60_FIXED_G1_SELECTION_DEVELOPMENT_SCOPE","human_message_verbatim":'Continue',
        "authorized_scope":R.expected_authorized_scope(),"execution_requires_frozen_reviewed_sources_contract_and_tests":True}


class BudgetAndAccessTests(unittest.TestCase):
    def test_member_aliases_cannot_reorder_or_replace_frozen_members(self):
        metadata={'debug':[{'image_id':'synthetic/debug'}],
            'selection':[{'image_id':'synthetic/selection_a'},{'image_id':'synthetic/selection_b'}],
            'calibration':{'synthetic_seed':[]},'prepared_root':'synthetic_metadata_root',
            'member_order_hashes':{'selection':'synthetic_hash'},'duplicate_policy':{'retain':True}}
        p=deepcopy(metadata);p['members_metadata']=deepcopy(metadata)
        R.validate_member_aliases(p)
        for field,value in [('selection',list(reversed(metadata['selection']))),('prepared_root','other_root'),
                            ('duplicate_policy',{'retain':False}),('debug',[]),('calibration',{}),('member_order_hashes',{})]:
            changed=deepcopy(p);changed[field]=value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):R.validate_member_aliases(changed)

    def test_exact_10713_forward_breakdown_and_raw_counts(self):
        b=R.budget();per=b['forward_breakdown']['per_seed']
        self.assertEqual(sum(per.values()),3352)
        self.assertEqual(548+3*sum(per.values()),10604)
        self.assertEqual(109+548+3*sum(per.values()),b['forward_calls'])
        self.assertEqual(b['forward_calls'],10713)
        self.assertEqual(b['unique_Selection_raw_files'],548+3*3*548)
        self.assertEqual(b['logical_Selection_arrays'],3*4*548)
        self.assertEqual(b['debug_preflight_raw_files'],109)
        self.assertEqual(b['timeout_seconds'],7200);self.assertEqual(b['attempts'],1)
        self.assertFalse(b['automatic_retry']);self.assertFalse(b['Holdout_access'])

    def test_authorization_exact_scope_not_self_generated(self):
        R.validate_authorization(authorization())
        for key,value in [('real_image_forward_budget',10714),('saved_submission_Python_AP_budget',11),
            ('calibration_labels_access',True),('Holdout','OPEN'),('automatic_retry',True),('candidate_count',2)]:
            receipt=authorization();receipt['authorized_scope'][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):R.validate_authorization(receipt)

    def test_human_decision_cannot_bypass_review_freeze(self):
        receipt=authorization();receipt['execution_requires_frozen_reviewed_sources_contract_and_tests']=False
        with self.assertRaises(ValueError):R.validate_authorization(receipt)

    def test_fixed_AP_call_order_shared_once_then_nine(self):
        jobs=R.AP_jobs()
        self.assertEqual(len(jobs),10);self.assertEqual(jobs[0],('fp32',None,'fp32'))
        self.assertEqual(len({job[0] for job in jobs}),10)
        self.assertEqual([job[0] for job in jobs],R.AP_policy()['order'])
        self.assertEqual([config for _,seed,config in jobs if seed=='2026102201'],R.CONFIGS[1:])
        self.assertTrue(R.AP_policy()['first500_across_classes_before_class_filter'])
        self.assertIsNone(R.AP_policy()['scientific_PASS_threshold'])

    def test_Selection_postprocessing_retains_numbers_with_correct_role(self):
        policy=R.postprocessing_policy();old=R.P.postprocessing_policy()
        self.assertEqual(policy['NMS'],old['NMS']);self.assertEqual(policy['restore'],old['restore'])
        self.assertEqual(policy['target_match'],old['target_match'])
        self.assertIn('Selection',policy['submission']);self.assertNotIn('debug',policy['submission'])
        self.assertEqual(policy['official_AP'],'SOURCE_FAITHFUL_PYTHON_SELECTION_DEVELOPMENT_10_CALLS')


class DebugGateTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'image_id':f'synthetic/debug_{index:03d}'} for index in range(109)]
        self.checks=[{'image_id':row['image_id'],'strict_equal':True} for row in self.rows]

    def test_exact_all_debug_evidence(self):
        evidence=R.debug_evidence(self.checks,self.rows,109)
        self.assertEqual(evidence['images'],109);self.assertEqual(evidence['Selection_content_reads_before_gate'],0)

    def test_shortened_or_wrong_budget_gate_is_rejected(self):
        for checks,rows,count in ((self.checks[:-1],self.rows,109),(self.checks,self.rows,108),(self.checks,self.rows,110)):
            with self.subTest(count=count,length=len(checks)):
                with self.assertRaises(ValueError):R.debug_evidence(checks,rows,count)

    def test_any_difference_or_identity_reorder_stops(self):
        for value in (False,1,None):
            checks=deepcopy(self.checks);checks[64]['strict_equal']=value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):R.debug_evidence(checks,self.rows,109)
        checks=deepcopy(self.checks);checks.reverse()
        with self.assertRaises(ValueError):R.debug_evidence(checks,self.rows,109)


class SavedInterchangeTests(unittest.TestCase):
    def test_original_GT_physical_rows_score0_category0_and_flags_preserved(self):
        text='\ufeff1,2,3,4,1,2,3,2\n\n5,6,7,8,0,0,0,0,\n9,10,11,12,0,3,1,1\n'
        payload=R.annotation_payload(text,'synthetic/member')
        self.assertEqual(payload['original_annotation_text'],text)
        self.assertEqual([row['physical_line_zero_based'] for row in payload['original_rows_with_physical_line']],[0,2,3])
        self.assertEqual(payload['raw_ground_truth_rows8'][1],[5.,6.,7.,8.,0.,0.,0.,0.])
        self.assertEqual(payload['parsed_project_ground_truth'][2]['target_id'],'synthetic/member:3')
        self.assertEqual(payload['parsed_project_ground_truth'][0]['occlusion'],2)

    def test_original_GT_invalid_nonfinite_or_flags_reject(self):
        for text in ('1,2,3,4,nan,1,0,0','1,2,3,4,1,1,0,8','1,2,-3,4,1,2,0,0','1,2,3,4,1,2,0'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):R.annotation_payload(text,'synthetic/member')

    def test_raw_submission_preserves_unclipped_coordinates_and_stable_ties(self):
        predictions=[{'bbox_xywh':[-4.,-2.,20.,30.],'score':.7,'category':1,'nms_input_order':0},
                     {'bbox_xywh':[1.,2.,3.,4.],'score':.7,'category':2,'nms_input_order':1}]
        rows=R.prediction_rows8(predictions)
        self.assertEqual(rows[0],[-4.,-2.,20.,30.,.7,1,0.,0.])
        with self.assertRaises(ValueError):R.prediction_rows8(list(reversed(predictions)))

    def test_synthetic_BGR_RGB_CHW_is_identical_contiguous_uint8(self):
        image=np.asarray([[[1,2,3],[4,5,6]]],dtype=np.uint8)
        result=R.color_chw(image)
        np.testing.assert_array_equal(result,np.asarray([[[3,6]],[[2,5]],[[1,4]]],dtype=np.uint8))
        self.assertEqual(result.dtype,np.uint8);self.assertTrue(result.flags.c_contiguous)
        for invalid in (image.astype(np.float32),image[...,0]):
            with self.assertRaises(ValueError):R.color_chw(invalid)

    def test_all_anchor_difference_keeps_large_low_score_error(self):
        reference=np.zeros((1,14,8400),dtype=np.float32);reference[:,2:4]=1
        candidate=reference.copy();candidate[0,0,-1]=5
        difference,summary=R.B.raw_difference(reference,candidate)
        self.assertEqual(difference.shape,(1,14,8400));self.assertEqual(difference.dtype,np.float64)
        self.assertEqual(summary['boxes']['changed_elements'],1);self.assertEqual(summary['boxes']['max_absolute_error'],5.)


if __name__=='__main__':unittest.main()
