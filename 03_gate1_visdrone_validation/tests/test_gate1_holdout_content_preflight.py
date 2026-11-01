"""Synthetic images/annotations and inventory metadata; no real archive access."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gate1_holdout_content_preflight as P


def inventory():
    records = []
    for index in range(1610):
        key = f'fabricated_{index:04d}'
        for role, folder, extension in [('image', 'images', 'jpg'), ('annotation', 'annotations', 'txt')]:
            records.append({'role': role, 'logical_id': key, 'archive_name': f'fake/{folder}/{key}.{extension}',
                'payload_opened': False, 'member_SHA256': None})
    return {'member_payload_reads': 0, 'filename_paired_image_annotation_count': 1610, 'records': records}


def png(orientation=None):
    from PIL import Image
    out = io.BytesIO()
    image = Image.new('RGB', (40, 30), (10, 20, 30))
    if orientation is None:
        image.save(out, format='PNG')
    else:
        exif = image.getexif(); exif[274] = orientation
        image.save(out, format='PNG', exif=exif)
    return out.getvalue()


class PreflightGuardTests(unittest.TestCase):
    def test_complete_fixed_order_and_pair_count(self):
        pairs = P.fixed_pairs(inventory())
        self.assertEqual(len(pairs), 1610)
        self.assertEqual(pairs[0]['image_id'], 'testdev/fabricated_0000')
        self.assertEqual(pairs[-1]['image_id'], 'testdev/fabricated_1609')

    def test_missing_annotation_stops_without_exclusion(self):
        value = inventory(); value['records'].pop()
        with self.assertRaises(ValueError): P.fixed_pairs(value)

    def test_duplicate_logical_identity_is_rejected(self):
        value = inventory(); value['records'].append(deepcopy(value['records'][0]))
        with self.assertRaises(ValueError): P.fixed_pairs(value)

    def test_member_path_escape_is_rejected(self):
        value = inventory(); value['records'][0]['archive_name'] = '../fake.jpg'
        with self.assertRaises(ValueError): P.fixed_pairs(value)

    def test_prior_payload_access_not_silently_assumed(self):
        value = inventory(); value['records'][0]['payload_opened'] = True
        with self.assertRaises(ValueError): P.fixed_pairs(value)

    def test_synthetic_image_and_original_annotation_identity(self):
        value = P.content_identity(png(), b'1,2,3,4,1,1,0,0\n')
        self.assertEqual((value['width'], value['height']), (40, 30))
        self.assertEqual(value['annotation_nonempty_rows'], 1)
        self.assertEqual(len(value['decoded_BGR_pixel_sha256']), 64)

    def test_nontrivial_EXIF_is_explicit_stop(self):
        with self.assertRaises(ValueError): P.content_identity(png(6), b'1,2,3,4,1,1,0,0\n')

    def test_bad_GT_and_nonfinite_geometry_rejected(self):
        for annotation in (b'1,2,-3,4,1,1,0,0', b'1,2,3,4,2,1,0,0', b'NaN,2,3,4,1,1,0,0'):
            with self.subTest(annotation=annotation):
                with self.assertRaises(ValueError): P.content_identity(png(), annotation)

    def test_same_pixels_conflicting_annotation_is_not_picked_or_merged(self):
        first = {'image_id': 'synthetic/a', 'annotation_physical_rows_sha256': 'a' * 64}
        second = {'image_id': 'synthetic/b', 'annotation_physical_rows_sha256': 'b' * 64}
        seen = {}; P.check_duplicate(seen, 'samepixels', first)
        with self.assertRaises(ValueError): P.check_duplicate(seen, 'samepixels', second)
        self.assertEqual(seen['samepixels'], first)

    def test_consistent_duplicate_retains_same_dependency_without_mutation(self):
        first = {'image_id': 'synthetic/a', 'annotation_physical_rows_sha256': 'a' * 64}
        second = {'image_id': 'synthetic/b', 'annotation_physical_rows_sha256': 'a' * 64}
        seen = {}; P.check_duplicate(seen, 'samepixels', first); P.check_duplicate(seen, 'samepixels', second)
        self.assertEqual(len(seen), 1)

    def test_inventory_consent_cannot_authorize_content_or_model(self):
        binding = {'path': 'synthetic/prepared', 'sha256': 'a' * 64, 'bytes': 1}
        decision = {'schema': 'rampq.gate1.Holdout_content_quality_human_decision/v1',
            'decision': 'AUTHORIZED_FIXED1610_CONTENT_QUALITY_ONLY', 'actual_access_authorized': True,
            'human_message_verbatim': 'synthetic fixture only', 'conversation_basis': 'synthetic',
            'scope': P.scope(), 'prepared': binding, 'further_model_scoring_access': False}
        P.validate_decision(decision, binding)
        for field, value in [('actual_access_authorized', False), ('decision', 'AUTHORIZED_CONTAINER_AND_CENTRAL_DIRECTORY_METADATA_ONLY'),
            ('further_model_scoring_access', True), ('human_message_verbatim', '')]:
            bad = deepcopy(decision); bad[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError): P.validate_decision(bad, binding)

    def test_budget_preserves_all_members_and_zero_models(self):
        value = P.budget()
        self.assertEqual(value['fixed_image_annotation_pairs'], 1610)
        self.assertEqual(value['model_calibration_NMS_matching_AP_calls'], 0)
        self.assertEqual(value['outer_container_hash_passes'], 2)
        self.assertFalse(value['automatic_retry'])
        self.assertFalse(P.scope()['automatic_member_exclusion_relabeling_or_choosing_duplicate_annotation'])


if __name__ == '__main__': unittest.main()
