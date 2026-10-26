"""CPU-only leakage, coverage, and failure-closed checks for study v2."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import analyze_sm120_length_study as study

class StudyTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'benchmarks/sm120_length_study_protocol.json').read_text())

    def test_fresh_shape_partition_and_matched_tokens(self):
        study.validate_protocol(self.protocol)
        old=json.loads((ROOT/'benchmarks/sm120_split_policy_protocol.json').read_text())
        shape=lambda c:tuple(c[k] for k in ('batch','sq','hq','hk','allocated_length'))
        self.assertFalse({shape(c) for c in old['cases']} & {shape(c) for c in self.protocol['cases']})
        self.assertFalse({c['seed'] for c in old['cases']} & {c['seed'] for c in self.protocol['cases']})
        bd=study.v1.bounds([c for c in self.protocol['cases'] if c['phase']=='train'])
        self.assertTrue(all(study.v1.outside(c,bd) is None for c in self.protocol['cases']))

    def test_reject_cross_phase_sibling(self):
        self.protocol['cases'][0]['phase']='holdout'
        with self.assertRaisesRegex(ValueError,'leakage'): study.validate_protocol(self.protocol)

    def test_reject_unmatched_total(self):
        self.protocol['cases'][2]['live_lengths'][0]+=1
        with self.assertRaisesRegex(ValueError,'Unmatched'): study.validate_protocol(self.protocol)

    def test_grouped_freeze_and_input_rejection(self):
        protocol=self.protocol
        cases=[c for c in protocol['cases'] if c['phase']=='train']
        data={c['id']:{s:[10.]*21 for s in c['candidate_splits']} for c in cases}
        raw={'source_hashes':{'runtime':'synthetic-test-only'},'environment':{},'completed_utc':'2026-10-26T00:00:00Z'}
        old_path=ROOT/'docs/data/sm120_split_policy_20261025/frozen_policy.json'
        args=type('Args',(),dict(protocol=str(ROOT/'benchmarks/sm120_length_study_protocol.json'),train='synthetic',previous=str(old_path),output='no-file-written'))()
        realsha=study.v1.sha
        with mock.patch.object(study,'parse',return_value=(data,raw)),mock.patch.object(study.v1,'write_new') as writer,mock.patch.object(study.v1,'sha',side_effect=lambda p:'synthetic' if p=='synthetic' or p=='no-file-written' else realsha(p)):
            study.freeze(args)
        frozen=writer.call_args.args[1]
        for trial in frozen['cross_validation']:
            self.assertEqual(len(trial['folds']),12)
            for fold in trial['folds']:
                self.assertNotIn(fold['held_family'],fold['training_families'])
                self.assertEqual(len(fold['training_families']),3)
        self.assertEqual(frozen['model']['family'],'default')
        for f in study.families(protocol['cases']):
            self.assertEqual(len({p['selected'] for p in frozen['predictions'] if p['family']==f}),1)

if __name__=='__main__': unittest.main()
