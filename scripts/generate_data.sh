#!/usr/bin/env bash
#
# Regenerate the benchmark dataset from scratch.
#
# IMPORTANT: this uses the Sparkov generator's *v0.5 release branch*, not master.
#
# The current master (v1.0) parallelises generation by chunking customers, and in
# doing so it walks spend categories sequentially through the simulated calendar
# instead of sampling them per transaction. The result is that every customer
# transacts in the same category at the same time: in a dataset we generated with
# it, `gas_transport` appeared only in Jan-Apr 2023 and `travel` only in Jun 2024,
# and in the training window `personal_care` contained 137 transactions of which
# 137 were fraud. Under a temporal split that makes `category` a near-perfect proxy
# for "which month is this", which inflates every metric and would have made this
# project's headline numbers meaningless.
#
# v0.5 is single-threaded per profile and does not have the defect: category shares
# are flat at roughly 1/14 in every month of the simulated period. It is slower, so
# the profiles are run concurrently here.
#
# Usage:  bash scripts/generate_data.sh [N_CUSTOMERS] [OUT_DIR]

set -euo pipefail

N_CUSTOMERS="${1:-700}"
OUT_DIR="${2:-$(pwd)/../raw05}"
SEED=4444
START="1-1-2023"
END="6-30-2024"
GEN_DIR="${GEN_DIR:-$(pwd)/../sparkov05}"

if [ ! -d "$GEN_DIR" ]; then
  git clone -b release/v0.5 \
    https://github.com/namebrandon/Sparkov_Data_Generation.git "$GEN_DIR"
fi

mkdir -p "$OUT_DIR"
cd "$GEN_DIR"

echo "generating $N_CUSTOMERS customers (seed $SEED)"
python3 datagen_customer.py "$N_CUSTOMERS" "$SEED" profiles/main_config.json \
  -o "$OUT_DIR/customers.csv"

PROFILES=(
  adults_2550_female_rural adults_2550_female_urban
  adults_2550_male_rural   adults_2550_male_urban
  adults_50up_female_rural adults_50up_female_urban
  adults_50up_male_rural   adults_50up_male_urban
  young_adults_female_rural young_adults_female_urban
  young_adults_male_rural   young_adults_male_urban
  leftovers
)

echo "generating transactions for ${#PROFILES[@]} profiles, $START -> $END"
for p in "${PROFILES[@]}"; do
  python3 datagen_transaction.py "$OUT_DIR/customers.csv" "profiles/$p.json" \
    "$START" "$END" -o "$OUT_DIR/$p.csv" &
done
wait

echo "done. raw files in $OUT_DIR"
ls -la "$OUT_DIR"
