# Replicate experimental results in the APTP paper

## Important Notes

- `APTPchecker` relies heavily on `Gurobi` whose academic license is freely available at [Request License](https://portal.gurobi.com/iam/licenses/request). 
- Throughout our experiments, we used `Gurobi` version 10 and installation scripts below all use Gurobi version 10. Please request correct version of Gurobi's license to run everything smoothly.
- Verifying problems and checking their proofs might take ***long time*** to finish. 
- We settled down with a timeout of ***1000 seconds per problem*** for every verifier in our experiments.
- Benchmarks comprised of 400 problems took ***more than 2 days*** to run a single verifier on our machine.
- Due to the size of pre-built Marabou is too large (~6.4G), we exclude it from this repository but people can easily follow the official [Marabou](https://github.com/NeuralNetworkVerification/Marabou)'s instruction to install. We provide this [script](verifier/marabou_proof/vnncomp_scripts/run_instance.sh) to match with our running scripts. After successfully installing Marabou, copy the [script](verifier/marabou_proof/vnncomp_scripts/run_instance.sh) into `vnncomp_scripts` folder in Marabou's main folder (e.g., Marabou would be installed at `verifier/marabou_proof/`) and it is all set.
- Due to anonymized reviewing process, we removed all the `logs` generated during experiments, as they might reveal our identities. These logs are required to plot our figures so in order to re-draw all the figures in the paper, please run a complete experiment.

## Hardware

- All experiments were run on a Linux machine with:
    - an AMD Threadripper 64-core 4.2GHZ CPU, 128GB RAM, 
    - an NVIDIA GeForce RTX 4090 GPU with 24 GB VRAM. 

- We used a timeout of 1000 seconds for the combined time of:
    - running the verifier 
    - generating the proof
    - checking that proof.

## Installation

Each verifier requires an unique working environment to work perfectly. These following commands will install `conda` environment for each verifier.

1. NeuralSAT: 

- Remove environment `benchmark-neuralsat` if exists:

    ```bash
    conda deactivate; conda env remove --name benchmark-neuralsat
    ```

- Install environment named `benchmark-neuralsat`:

    ```bash
    conda env create -f installation/neuralsat.yaml
    ```

2. alpha-beta-CROWN: 

- Remove environment `benchmark-abcrown` if exists:

    ```bash
    conda deactivate; conda env remove --name benchmark-abcrown
    ```

- Install environment named `benchmark-abcrown`:

    ```bash
    conda env create -f installation/abcrown.yaml
    ```


3. Marabou: 

- Remove environment `benchmark-marabou` if exists:

    ```bash
    conda deactivate; conda env remove --name benchmark-marabou
    ```

- Install environment named `benchmark-marabou`:

    ```bash
    conda env create -f installation/marabou.yaml
    ```

- Note that due to the size of pre-built Marabou is too large (~6.4G), we exclude it from this repository but people can easily follow the official [Marabou](https://github.com/NeuralNetworkVerification/Marabou)'s instruction to install. We provide this [script](verifier/marabou_proof/vnncomp_scripts/run_instance.sh) to match with our running scripts. After successfully installing Marabou, copy the [script](verifier/marabou_proof/vnncomp_scripts/run_instance.sh) into `vnncomp_scripts` folder in Marabou's main folder and it is all set.

## Run experiments

1. NeuralSAT: 

- Activate `conda` environment

    ```bash
    conda activate benchmark-neuralsat
    ```

- Change directory into `NeuralSAT` folder

    ```bash
    cd verifier/neuralsat_proof/
    ```

- Run experiments

    ```bash
    NeuralSAT_S=0 X=0 S=0 python3 ../../runner.py --tool neuralsat_default
    ```

    ```bash
    NeuralSAT_S=0 X=1 S=0 python3 ../../runner.py --tool neuralsat_X
    ```

    ```bash
    NeuralSAT_S=0 X=0 S=1 python3 ../../runner.py --tool neuralsat_S
    ```

    ```bash
    NeuralSAT_S=0 X=1 S=1 python3 ../../runner.py --tool neuralsat_SX
    ```

    ```bash
    NeuralSAT_S=1 X=1 S=1 python3 ../../runner.py --tool neuralsat_S_SX
    ```


2. alpha-beta-CROWN: 

- Activate `conda` environment

    ```bash
    conda activate benchmark-abcrown
    ```

- Change directory into `alpha-beta-CROWN` folder

    ```bash
    cd verifier/abcrown_proof/
    ```

- Run experiments

    ```bash
    ABCROWN_PROOF_EXP_CONFIG=./complete_verifier/proof_config/default.yaml X=1 S=1 python3 ../../runner.py --tool abcrown_default
    ```

    ```bash
    ABCROWN_PROOF_EXP_CONFIG=./complete_verifier/proof_config/babsr.yaml X=1 S=1 python3 ../../runner.py --tool abcrown_babsr
    ```

3. Marabou: 

- Activate `conda` environment

    ```bash
    conda activate benchmark-marabou
    ```


- Change directory into `Marabou` folder (suppose that `Marabou` has been already installed at `verifier/marabou_proof/` and `vnncomp_scripts/` folder is already copied there)

    ```bash
    cd verifier/marabou_proof/
    ```

- Run experiments

    ```bash
    PROVE_UNSAT=True python3 ../../runner.py --tool marabou
    ```

    ```bash
    PROVE_UNSAT=False python3 ../../runner.py --tool marabou_verify
    ```


## Plot results


- Activate `conda` environment

    ```bash
    conda activate benchmark-neuralsat
    ```

- Change directory into `plots` folder

    ```bash
    cd plots/
    ```

- Run plots

    ```bash
    ./run_plot.sh 
    ```
