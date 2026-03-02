#!/bin/bash
set -euo pipefail

BASE_DIR="${1:-.}"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval"

SKIP_DATASETS="mmvet infovqa stvqa"

should_skip() {
    local ds="$1"
    for skip in $SKIP_DATASETS; do
        if [ "$ds" = "$skip" ]; then
            return 0
        fi
    done
    return 1
}

for answers_root in "${BASE_DIR}"/answers_*/; do
  [ -d "$answers_root" ] || continue
  label=$(basename "$answers_root")

  for ds_dir in "${answers_root}"/*/; do
    [ -d "$ds_dir" ] || continue
    ds=$(basename "$ds_dir")

    if should_skip "$ds"; then
      echo "Skipping ${ds} (${label})"
      continue
    fi

    test_script="${EVAL_DIR}/${ds}/${ds}_test.py"

    if [ ! -f "$test_script" ]; then
      echo "Skipping ${ds} for ${label}: no test script at ${test_script}"
      continue
    fi

    echo "Running ${ds}_test.py --compare_dir ${ds_dir} (${label})"
    python "$test_script" --compare_dir "$ds_dir"
  done
done

echo "All test scripts completed."
