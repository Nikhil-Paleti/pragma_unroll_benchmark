#!/bin/bash
ncu --metrics \
  smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg,\
smsp__pcsamp_warps_issue_stalled_wait.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed.avg,\
sm__cycles_elapsed.avg \
  --kernel-name regex:kernel_unroll.* \
  --launch-count 1 \
  ./benchmark 2>&1 | tee ncu_output.txt
