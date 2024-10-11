#! /usr/bin/env bash

version=$1
benchmark=$2
onnx=$3
vnnlib=$4
result=$5
timeout=$6

pkill -9 Marabou
# pkill -9 python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ "$PROVE_UNSAT" = "True" ]; then
    echo "Running Verify + Prove"
    python3 "$SCRIPT_DIR"/../resources/runMarabou.py $onnx $vnnlib --summary-file=$result --timeout=$timeout --verbosity=2 --num-workers=64 --prove-unsat
else
    echo "Running Verify only"
    python3 "$SCRIPT_DIR"/../resources/runMarabou.py $onnx $vnnlib --summary-file=$result --timeout=$timeout --verbosity=2 --num-workers=64
fi
