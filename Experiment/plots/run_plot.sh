# Fig 4
python3 plot_results.py --type fnn --include_verify --format pdf
python3 plot_results.py --type cnn --include_verify --format pdf

# Fig 5
python3 plot_hist_sub_proof.py --verifier neuralsat_default --n_bins 35 --format pdf
python3 plot_hist_sub_proof.py --verifier neuralsat_SX --n_bins 4 --format pdf --compact
python3 plot_hist_milp_complexity.py --verifier neuralsat_default --format pdf --n_bins 30 --postfix NONE
python3 plot_hist_milp_complexity.py --verifier neuralsat_SX --format pdf --n_bins 21 --postfix SX

# Fig 6
python3 plot_results_ablation.py --type '' --format pdf