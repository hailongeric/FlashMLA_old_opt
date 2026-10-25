# Raw evidence for the 25 October 2026 SM120 study

See [the interpretation](../../SM120_SPLIT_POLICY_RESULTS.md) and [frozen method](../../SM120_SPLIT_POLICY_METHOD.md). Raw results and the fitted-policy record are byte-for-byte copies of server outputs. Original 24 October evidence remains in its existing files. The preregistration JSON reproduces the recorded metadata, source hashes and timestamp from the server audit.

Files prefixed `replication_` use the previous timing order and retain that caveat. Files prefixed `balanced_` use the corrected helper (20 rounds forward, 16 synthetic). `train_A` is the sole model-fit input. `session_B` and `session_C` are independent fresh-process confirmations of the same frozen choices. `evaluation` is produced by the frozen CPU analyzer without refitting. GPU guards are sampled observations, not reservations.

| File | Bytes | SHA256 |
|---|---:|---|
| [balanced_low_batch.json](balanced_low_batch.json) | 145121 | `f9973d7cce41280a1b0d39b319868129963d8217e4cddda8877a6c9f980d092d` |
| [balanced_split_grid.json](balanced_split_grid.json) | 65709 | `18fc27f4b06e97d9770e7be1dde7c406b7e8af8aac82e137619103d377033cdf` |
| [evaluation.json](evaluation.json) | 44230 | `21a4124b99b19f91bd5869a7c5d8fcf1f84d1743742e02a4f620a309c5b5267f` |
| [frozen_policy.json](frozen_policy.json) | 39504 | `005fed5ad030f2f3cb891b7bb0942a5257a068092b8decf4adaedd8544c4678d` |
| [preregistration.json](preregistration.json) | 881 | `fa4f19abc89a91bcf635e55c7cbcb6a297110b093d59412d215480afce891dcc` |
| [replication_low_batch.json](replication_low_batch.json) | 121213 | `d60c571cae9c40010b5c29226278ed053e9c2d1e7d2f5b7fabe27269995fea7a` |
| [replication_split_grid.json](replication_split_grid.json) | 63715 | `aff2de90467335f9633e4367615fe47173f0cd4a594f76a602c5e8d6e1518ce9` |
| [session_B.json](session_B.json) | 2663844 | `547e1f5dc0ca672d6fd3b16c6fe4ccedb2dc33f1e2177ee157d2b0fa199ff183` |
| [session_C.json](session_C.json) | 2663709 | `dfc26fb3949091573c753a63cdcc200a50da3ac80fa8e71d2d0d7d3cfcbdc321` |
| [train_A.json](train_A.json) | 1568781 | `9d9a38673a1d007794e1f5067e580d6169c4defb618a8f4666ce799714144161` |
