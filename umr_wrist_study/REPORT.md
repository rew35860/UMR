# UMR scaled-box wrist investigation

Sequence: sub10_smallbox_084. Existing local revision: fcdce48 (no update pulled). 166 frames at 30 Hz, cached correspondence, standard HOI defaults and G1 config, source/box scale 0.8037967. All 12 experiments completed. Original motion and previous results were preserved; generated per-run floating MJCF files are in UMR/assets/g1.

## Conclusion

The main trigger is an abrupt rotation already present in the original OMOMO source motion. Changing the random seed changes the robot trajectory but did not resolve this twist in the tested seeds.

The converted source body rotations match the original OMOMO training-split entry exactly after float32 conversion. The female SMPL-X model parent table confirms the left-hand chain 0,3,6,9,13,16,18,20. Composing those rotations gives actual world-orientation jumps of 160.377 degrees between zero-based frames 60–61 (2.000–2.033 seconds), and 164.799 degrees between frames 71–72 (2.367–2.400 seconds). The forearm also jumps about 46.5 and 47.9 degrees. These measurements use relative rotations, so they are not an angle-display wrap. The reason the original recording/body fit contains these jumps is not established; checking original footage would help distinguish real movement from fitting errors.

UMR follows source surface positions and normals. The discontinuity changes these targets abruptly; sequential initialization, regularization and filtering spread the response over multiple frames. The robot's left wrist roll reaches its model limit of -1.97222 radians (-113 degrees).

## Experiments

Three independent seed-0 runs produced exactly identical qpos arrays, also exactly matching the original scaled-box output. Solver seeds 1, 2, 3 changed same-frame wrist angles by up to 63.0, 107.2, 63.0 degrees respectively relative to seed 0. All still reached -113 degrees of left roll. The solver seed samples correspondence points, including 15 per hand. Dataset generation and correspondence training have separate seeds (default 0); retraining was not tested. Repeatability is established for this installed retargeting setup, not all hardware or training configurations.

| Variant | Minimum left roll, degrees | Maximum frame step across six wrist joints, degrees |
|---|---:|---:|
| Seed 0 (three identical runs) | -113.0 | 17.94 |
| Seed 1 | -113.0 | 17.98 |
| Seed 2 | -113.0 | 14.86 |
| Seed 3 | -113.0 | 17.90 |
| Bidirectional initialization | -113.0 | 17.94 |
| Disable box collision constraint | -113.0 | 18.00 |
| Hand normal weight 1 to 5 | -113.0 | 17.44 |
| All available hand points | -113.0 | 17.86 |
| Interpolate source left wrist | -87.6 | 16.31 |
| Interpolate source left elbow and wrist | -56.4 | 7.40 |

Interpolation experiments modify copies only: replace frames 61–71 inclusive using shortest-path quaternion interpolation between frames 60 and 72 for joint 20, or joints 18 and 20. All other motion, object trajectory, solver settings and correspondence stay the same. Combined elbow/wrist interpolation eliminates the roll-limit hit and reduces the maximum wrist step about 59%, providing strong causal evidence for the source discontinuity. This is a diagnostic repair, not recovered ground truth. Contact accuracy and overall fidelity of the repaired versions have not been validated; remaining wrist movement has not been classified as erroneous.

Hand point/normal weights are 10/1, forearm normal weight is 0, object contact-map cost is 0, and box penetration constraints use slack cost 5000. These affect the result, but the controlled tests do not support blaming collision or sampling alone for the main twist. Stronger orientation fitting can also follow a bad source orientation more strongly.

## Files and reproduction

- wrist_diagnosis.png: source jumps, seed comparison and diagnostic repair comparison.
- metrics.csv / metrics.json: all 12 results; timings.jsonl: wall times (about 15–23 seconds each with cached correspondence).
- source_verification.json: raw-data match and model parent table.
- seed0_a.npz: reproduced baseline; source_elbow_wrist_slerp.npz: diagnostic repair.
- Each output has a log; seed and solver variants have JSON defaults.
- run_followups.py and test_source.py reproduce diagnostics. analyze.py produces measurements and plots using the existing hssim Python environment.

Logged cost means are implementation-specific weighted residual means before the final QP update/filtering, not wrist-quality scores; changing weights/samples changes their meaning.

From UMR, reproduce a seed run with a fresh output path:

```bash
/home/natcha/miniconda3/envs/umr/bin/python scripts/humanoid_retarget_pipeline_hsi_hoi.py --defaults /home/natcha/Downloads/umr_wrist_study/seed0_a.json --data sample_data/omomo --seq-key sub10_smallbox_084 --stage retarget --skip-view --out /home/natcha/Downloads/umr_wrist_study/reproduction.npz
```

Use seed1.json, seed2.json, seed3.json for other seeds. Existing output paths can be reused by UMR; add --force-retarget to rerun an existing path deliberately.

Paper context: https://arxiv.org/html/2609.02134 describes surface position/orientation matching with reusable correspondence. The sequence-specific findings here come from local experiments, not claims in the paper.
