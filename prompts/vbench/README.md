# VBench prompts for sampling

VBench's standard mode looks a video up by the file name `<official short prompt>-<i>.mp4`, but this model is trained
on long, LLM-extended prompts only (median 580 characters; the official VBench prompts have a median of 30). So the
videos are generated from the extended prompt and named after the short one, as the official Self-Forcing protocol does.
Line `i` of an `*_extended.txt` file belongs to line `i` of its short file.

| file | lines | what |
|---|---|---|
| `all_dimension_unique.txt` | 944 | the 946 official VBench prompts (`prompts/all_dimension.txt` of Vchitect/VBench, commit `fd18b3d`) with the 2 repeated lines dropped, first occurrence kept. The repeated prompts have two different extended versions upstream, so keeping both would let a race between GPUs decide which one a video shows. |
| `all_dimension_unique_extended.txt` | 944 | the matching lines of `prompts/vbench/all_dimension_extended.txt` of guandeh17/Self-Forcing, commit `33593df3e81fa3ec10239271dd2c100facac6de1` |
| `temporal_flickering.txt` | 75 | the temporal-flickering prompts (`prompts/prompts_per_dimension/temporal_flickering.txt` of VBench), for the 25-sample pass the static filter needs |
| `temporal_flickering_extended.txt` | 75 | their extended versions, same mapping |

Used by `scripts/eval/sample_for_vbench.sh` (suites `short` and `short_flicker`).
