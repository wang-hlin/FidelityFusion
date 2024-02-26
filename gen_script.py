import os

BENCHMARKS = {"fcnet": ["naval", "parkinsons", "protein", "slice"],
              "lcbench": ["airlines", "albert", "christine", "covertype", "fashion-mnist"],
              "nas201": ["cifar10", "cifar100", "image"]}

AR_MODEL = ["AR", "CIGAR", "GAR", "CAR", "NAR", "ResGP", "CIGP", "HOGP"]

root = "C:\\Data\\automl_benchmark\\processed"  # data root path on HPC
