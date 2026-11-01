"""BytesIO fixtures only; no actual archive/member data is accessed."""
from copy import deepcopy
import io
from pathlib import Path
import stat
import struct
import sys
from unittest.mock import patch
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_holdout_inventory as I


def make_zip(entries=None):
    stream = io.BytesIO()
    if entries is None:
        entries = [("synthetic/images/fake_1.jpg", b"fabricated-not-an-image"),
                   ("synthetic/annotations/fake_1.txt", b"fabricated-not-a-label")]
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return stream.getvalue()


def inspect_without_payload(data, limits=None):
    with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("member payload open forbidden")), \
         patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("member payload read forbidden")), \
         patch.object(zipfile.ZipFile, "extract", side_effect=AssertionError("extract forbidden")), \
         patch.object(zipfile.ZipFile, "extractall", side_effect=AssertionError("extractall forbidden")), \
         patch.object(zipfile.ZipFile, "testzip", side_effect=AssertionError("payload integrity scan forbidden")):
        return I.central_inventory(io.BytesIO(data), len(data), limits=limits)


def test_infolist_only_succeeds_with_all_payload_methods_forbidden_and_SHA_unknown():
    result = inspect_without_payload(make_zip())
    assert result["counts_by_role"] == {"annotation": 1, "image": 1}
    assert result["filename_paired_image_annotation_count"] == 1
    assert result["member_payload_reads"] == 0 and result["member_SHA256_available"] is False
    assert all(row["member_SHA256"] is None and row["payload_opened"] is False for row in result["records"])
    assert result["actual_scene_or_content_leakage_or_scientific_sufficiency"] == "UNKNOWN"
    assert all(len(row["CRC32_hex"]) == 8 for row in result["records"])


def test_deterministic_sorted_central_metadata_and_order_hash():
    entries = [("synthetic/images/b.jpg", b"b"), ("synthetic/annotations/b.txt", b"b-label"),
               ("synthetic/images/a.jpg", b"a"), ("synthetic/annotations/a.txt", b"a-label")]
    first = inspect_without_payload(make_zip(entries))
    second = inspect_without_payload(make_zip(entries))
    assert first == second and first["record_count"] == 4
    names = [row["normalized_name"] for row in first["records"]]
    assert names == sorted(names)


@pytest.mark.parametrize("unsafe", ["../escape.jpg", "/absolute.jpg", "[OMITTED_LOCAL_PATH]", "safe/a.jpg:stream",
    "safe\\backslash.jpg", "a/../b.jpg", "safe//empty.jpg", "safe/CON.txt", "safe/trailing. "])
def test_unsafe_windows_or_zip_path_names_rejected(unsafe):
    with pytest.raises(I.InventoryError):
        inspect_without_payload(make_zip([(unsafe, b"synthetic")]))


def test_symlink_special_and_NUL_metadata_names_rejected():
    symlink = zipfile.ZipInfo("synthetic/images/link.jpg")
    symlink.create_system = 3; symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(I.InventoryError):
        inspect_without_payload(make_zip([(symlink, b"fabricated-target")]))
    nul = zipfile.ZipInfo("synthetic/NUL\x00trailing")
    with pytest.raises(I.InventoryError):
        I.safe_member_name(nul)


@pytest.mark.parametrize("change", ["encrypted", "unsupported_compression", "huge_uncompressed", "invalid_offset"])
def test_unsafe_compressed_central_directory_metadata_rejected_without_payload(change):
    data = bytearray(make_zip()); position = data.index(b"PK\x01\x02")
    if change == "encrypted":
        flags = struct.unpack_from("<H", data, position + 8)[0]
        struct.pack_into("<H", data, position + 8, flags | 1)
    elif change == "unsupported_compression": struct.pack_into("<H", data, position + 10, 99)
    elif change == "huge_uncompressed": struct.pack_into("<I", data, position + 24, 0x7fffffff)
    else: struct.pack_into("<I", data, position + 42, 0x7fffffff)
    with pytest.raises(I.InventoryError):
        inspect_without_payload(bytes(data))


def test_duplicate_casefold_and_filename_pair_conflicts_rejected():
    for entries in ([('synthetic/images/a.jpg', b'a')],
                    [('synthetic/images/a.jpg', b'a'), ('synthetic/images/A.jpg', b'b'), ('synthetic/annotations/a.txt', b'label')],
                    [('first/images/a.jpg', b'a'), ('second/images/a.png', b'b'), ('synthetic/annotations/a.txt', b'label')]):
        with pytest.raises(I.InventoryError):
            inspect_without_payload(make_zip(entries))


def test_entry_and_metadata_size_caps_are_real_stops():
    limits = I.budget(); limits['maximum_entries'] = 1
    with pytest.raises(I.InventoryError): inspect_without_payload(make_zip(), limits)
    limits = I.budget(); limits['maximum_inventory_metadata_bytes'] = 16
    with pytest.raises(I.InventoryError): inspect_without_payload(make_zip(), limits)


def test_prepare_contract_is_pure_and_never_probes_archive_or_D_destination():
    seal = {'status':'SEALED_OPAQUE_ARCHIVE_NOT_EXTRACTED', 'archive':{'path':'[OMITTED_LOCAL_PATH]',
        'bytes':I.EXPECTED_ARCHIVE_BYTES,'sha256':I.EXPECTED_ARCHIVE_SHA}}
    probe = {'status':'PASS_SINGLE_NEW_D_DIRECTORY_GENERATED_1MiB_WRITE_READ_ONLY', 'new_absolute_directory':'[OMITTED_LOCAL_PATH]'}
    with patch.object(Path,'open',side_effect=AssertionError('actual path open forbidden')), \
         patch.object(Path,'exists',side_effect=AssertionError('archive probe forbidden')), \
         patch.object(Path,'stat',side_effect=AssertionError('archive stat forbidden')), \
         patch.object(Path,'resolve',side_effect=AssertionError('actual path resolution forbidden')):
        contract = I.build_prepared(seal,probe,{'fabricated_binding':{'sha256':'f'*64}})
    assert contract['human_access_authorized'] is False and contract['archive_body_currently_probed'] is False
    assert contract['current_Holdout_member_count'] is None
    assert contract['budget']['model_forwards']==contract['budget']['member_payload_reads']==0
    assert '[OMITTED_LOCAL_PATH]' not in str(contract)


def test_pending_or_escalated_authorization_rejected_even_with_supplied_scope():
    binding={'path':'synthetic/prepared.json','bytes':1,'sha256':'a'*64}
    good={'schema':'rampq.gate1.Holdout_inventory_human_access_decision/v1',
        'decision':'AUTHORIZED_CONTAINER_AND_CENTRAL_DIRECTORY_METADATA_ONLY','actual_access_authorized':True,
        'human_message_verbatim':'synthetic fixture, no real authorization','conversation_basis':'fabricated test only',
        'scope':I.exact_scope(),'prepared_contract':binding,
        'execution_requires_source_review_and_new_frozen_authorized_contract':True,
        'further_image_label_content_or_inference_access':False}
    I.validate_human_decision(good,binding)
    for field,value in [('actual_access_authorized',False),('decision','PENDING_HUMAN_DECISION'),
                        ('further_image_label_content_or_inference_access',True),('human_message_verbatim',None)]:
        bad=deepcopy(good);bad[field]=value
        with pytest.raises(I.InventoryError):I.validate_human_decision(bad,binding)
    bad=deepcopy(good);bad['scope']['ZIP_member_payload_open_read_test_or_extract']=True
    with pytest.raises(I.InventoryError):I.validate_human_decision(bad,binding)
