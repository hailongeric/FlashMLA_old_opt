# Reviewed raw data: large-row and sequence-length study

Immutable JSON copies from the preregistered 26 October study. See the
[method](../../SM120_LENGTH_STUDY_METHOD.md), [results](../../SM120_LENGTH_STUDY_RESULTS.md)
and [workflow](../../WORKFLOW.md#9-reproduce-the-larger-row-and-matched-length-study).

| File | SHA256 |
|---|---|
| [preregistration.json](preregistration.json) | `e4d3e258ccc626b546442b18e2a3b7667c4d31513e39c5a320db76734f5c27cc` |
| [train_A.json](train_A.json) | `ab4f42e1114a36eef24b2ca787fb06cb145bddf6bdfbe4c72d1d8b5e6893dde4` |
| [train_A_guards.json](train_A_guards.json) | `84dae1578c6abd01daf10b0059fcb416b5bcb9f150e8fb7df958ec2e3f6119ae` |
| [frozen_policy.json](frozen_policy.json) | `e24b35edaf7e39bba9ac8c6dda24f8ced36399671ed4dde4b36d82d047ae8fb6` |
| [session_B.json](session_B.json) | `ab74f8586fa569c7259faa3a059010806aed4184a8ac7323bbb5b3fbebac9ba7` |
| [session_B_guards.json](session_B_guards.json) | `0777ff2a14e4acb74b6563e9dff7d96ca04a99e83b860509baeffcda74a3d7f1` |
| [session_C.json](session_C.json) | `4cc4067b85b5e3498a2b43b52f2e1f8f592ac070d340a160d4a28a302bf7eec9` |
| [session_C_guards.json](session_C_guards.json) | `26ea7d3e6511165db3c7c14630c20394afbaa6dc6917d2d1388e5b5dbcd1294a` |
| [evaluation.json](evaluation.json) | `183602c41431e79d2d54e17cc176131c2ecd9e6d021ddf9da9535f4679b63bed` |

`train_A` has only 12 training cases; B/C have only 12 held-out cases each.
The three matched profiles per family are kept together in CV and bootstrap.
Guard records contain periodic compute-process observations. Raw results include
source identities, frozen policy identity, numerical checks, environment, timing
order and every per-round stage sample. No timings or failures are filtered.
The prior frozen policy is preserved in the [v1 archive](../sm120_split_policy_20261025/frozen_policy.json).
