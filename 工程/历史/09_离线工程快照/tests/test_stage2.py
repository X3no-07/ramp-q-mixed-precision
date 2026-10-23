"""Check evidence corruption, path containment, and numeric failure handling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('stage2', Path(__file__).resolve().parents[1] / 'src/stage2.py')
stage2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage2)


class EvidenceTests(unittest.TestCase):
    def test_changed_evidence_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            payload = root / 'evidence.txt'
            payload.write_bytes(b'original')
            info = {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest(),
                    'source_relative_path': 'evidence.txt'}
            (root / 'config/source_manifest.json').write_text(json.dumps({'files': {'evidence.txt': info}}))
            self.assertEqual(stage2.verify(root)['status'], 'VERIFIED')
            payload.write_bytes(b'modified')
            self.assertEqual(stage2.verify(root)['status'], 'FAILED')

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                stage2.inside(Path(folder), '../outside')

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            path.write_text('original')
            with self.assertRaises(FileExistsError):
                stage2.save_or_print({}, path)
            self.assertEqual(path.read_text(), 'original')

    def test_numeric_contract(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy not installed')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'arrays.npz'
            for array in (np.array([np.nan], dtype='float32'), np.ones(2, dtype='float32'),
                          np.ones(1, dtype='float64')):
                np.savez(path, x=array)
                with self.assertRaises(ValueError):
                    stage2.load_output(path, 'x', [1])
            np.savez(path, x=np.ones(1, dtype='float32'))
            with self.assertRaises(ValueError):
                stage2.load_output(path, 'missing', [1])


if __name__ == '__main__':
    unittest.main()
