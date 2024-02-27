import os
import numpy as np

BENCHMARKS = {"fcnet": ["naval", "parkinsons", "protein", "slice"],
              "lcbench": ["airlines", "albert", "christine", "covertype", "fashion-mnist"],
              "nas201": ["cifar10", "cifar100", "image"]}

# AR_MODEL = ["AR", "CIGAR", "GAR", "CAR", "NAR", "ResGP", "CIGP", "HOGP"]
AR_MODEL = ["AR", "CIGAR",  "CAR", "ResGP"]

SEEDS = 2025 - np.arange(5)

# TRAIN_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05]
TRAIN_RATIOS = [0.05]

data_root = "/mnt/parscratch/users/ac1sz/data/automl_benchmark"  # data root path on HPC
output_root = "/mnt/parscratch/users/ac1sz/output/automl"  # output root path on HPC
batch_root = "/users/ac1sz/batch/automl"  # batch root path on HPC
cfg_root = "/users/ac1sz/cfgs/automl"  # cfg root path on HPC
code_root = "/users/ac1sz/code/FidelityFusion"  # code root path on HPC


def mk_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def main():
    for benchmark in BENCHMARKS:
        for task_name in BENCHMARKS[benchmark]:
            cfg_dir = os.path.join(cfg_root, "%s_%s" % (benchmark, task_name))
            mk_dir(cfg_dir)
            batch_dir = os.path.join(batch_root, "%s/%s/batch" % (benchmark, task_name))
            mk_dir(batch_dir)
            sbatch_fname = "%s_%s.sh" % (benchmark, task_name)
            sbatch_fpath = os.path.join(batch_dir, sbatch_fname)
            sbatch_file = open(sbatch_fpath, "w")
            for model in AR_MODEL:
                for seed in SEEDS:
                    for train_ratio in TRAIN_RATIOS:
                        train_ratio_str = str(train_ratio).replace(".", "_")
                        fname_prefix = f"{benchmark}_{task_name}_{model}_{seed}_{train_ratio_str}"
                        output_dir = os.path.join(output_root, fname_prefix)
                        mk_dir(output_dir)

                        cfg_fname = "%s.yaml" % fname_prefix
                        cfg_fpath = os.path.join(cfg_dir, cfg_fname)
                        cfg_file = open(cfg_fpath, "w")
                        cfg_file.write("DATASET:\n")
                        cfg_file.write("  BENCHMARK: %s\n" % benchmark)
                        cfg_file.write("  TASK: %s\n" % task_name)
                        cfg_file.write("  TRAIN_RATIO: %s\n" % train_ratio)
                        cfg_file.write("MODEL:\n")
                        cfg_file.write("  NAME: %s\n" % model)
                        cfg_file.write("SOLVER:\n")
                        cfg_file.write("  SEED: %s\n" % seed)
                        cfg_file.write("OUTPUT:\n")
                        cfg_file.write("  ROOT: %s\n" % output_dir)
                        cfg_file.close()

                        batch_fname = "%s.sh" % fname_prefix
                        batch_fpath = os.path.join(batch_dir, batch_fname)
                        batch_file = open(batch_fpath, "w")
                        batch_file.write("#!/bin/bash\n")
                        batch_file.write("# SBATCH --mem=10G\n")
                        batch_file.write("# SBATCH --output=%s/%s.txt\n" % (output_dir, fname_prefix))
                        batch_file.write("\n")
                        batch_file.write("module load Anaconda3/2022.05\n")
                        batch_file.write("source activate torch\n")
                        batch_file.write("\n")
                        batch_file.write("cd %s\n" % code_root)
                        batch_file.write("python main.py --cfg %s\n" % cfg_fpath)
                        batch_file.close()

                        sbatch_file.write("sbatch %s\n" % batch_fpath)
            sbatch_file.close()


if __name__ == "__main__":
    main()
