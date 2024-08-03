#!/bin/bash

VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo
echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))
echo $TOOL_DIR

ABCROWN_PROOF_FILE=${TOOL_DIR}/complete_verifier/generate_proof.py
# ABCROWN_PROOF_EXP_CONFIG=${TOOL_DIR}/complete_verifier/proof_config/default.yaml
# ABCROWN_PROOF_EXP_CONFIG=${TOOL_DIR}/complete_verifier/proof_config/random_decision.yaml
# ABCROWN_PROOF_EXP_CONFIG=${TOOL_DIR}/complete_verifier/proof_config/no_beta.yaml
# ABCROWN_PROOF_EXP_CONFIG=${TOOL_DIR}/complete_verifier/proof_config/babsr.yaml
echo "Configuration: '$ABCROWN_PROOF_EXP_CONFIG'"


export PYTHONPATH=${TOOL_DIR}
export OMP_NUM_THREADS=1

# run the tool to produce the results file
# python3 ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT"

python3 $ABCROWN_PROOF_FILE --config "$ABCROWN_PROOF_EXP_CONFIG" --onnx_path "$ONNX_FILE" --vnnlib_path "$VNNLIB_FILE" --timeout "$TIMEOUT" --results_file "$RESULTS_FILE"
EXIT_CODE=$?
echo "exit code: ${EXIT_CODE}"
