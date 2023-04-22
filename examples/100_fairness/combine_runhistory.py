from collections import defaultdict
import json
import os
import shutil
from run_experiment import run_experiment
import argparse, sys
import copy
parser=argparse.ArgumentParser()
from jsonmerge import Merger
parser.add_argument("--constrains", type= str, nargs="*", default=[])
parser.add_argument("--methods",  type= str, nargs="*", default=[])
parser.add_argument("--tmp", type=str,)
parser.add_argument("--runtime", type=str)
args=parser.parse_args()


constrains = args.constrains 
methods = args.methods
filepath = args.tmp
runtime = args.runtime
data = defaultdict()
for constrain in constrains:
        data[constrain] = defaultdict()
        constrain_path = "{}{}".format(filepath, constrain)
        for dataset in os.listdir(constrain_path):
            data[constrain][dataset] = defaultdict()
            dataset_path = "{}/{}".format(constrain_path, dataset)
            for seed in os.listdir(dataset_path):
                data[constrain][dataset][seed] = defaultdict()
                seed_path = "{}/{}".format(dataset_path, seed)
                for method in methods:
                    method_path = "{}/{}/{}".format(seed_path, method, runtime)
                    data[constrain][dataset][seed][method] = defaultdict()
                    data[constrain][dataset][seed][method]["points"] = []
                    data[constrain][dataset][seed][method]["configs"] = []
                    
                    runetime_folder = runtime
                    for rf_seed in os.listdir(method_path):
                        file_g  = "{}/{}/runhistory.json".format(method_path,rf_seed)
                        if os.path.exists(file_g):
                            continue
                        file = "{}/{}/{}/{}/del/smac3-output/run_{}/runhistory.json".format(seed_path,method,runetime_folder,rf_seed,seed)
                        with open(file) as f:
                            f1 = json.load(f)
                        file = "{}/{}/{}/{}/2t/del/smac3-output/run_{}/runhistory.json".format(seed_path,method,runetime_folder,rf_seed, seed)
                        nn = len(f1["configs"]) + 1
                        with open(file) as f:
                            f2 = json.load(f)
                            f2["data"].pop(0)
                            f2["config_origins"].pop("1")
                            f2["configs"].pop("1")
                            help_origns = defaultdict()
                            help_configs = defaultdict()
                            for id, d in enumerate(f2["data"]):
                                if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                                    continue
                                help_configs[str(nn)] =  f2["configs"][str(d[0][0])]
                                help_origns[str(nn)] =  f2["config_origins"][str(d[0][0])]
                                f2["data"][id][0][0] = nn
                                nn +=1
                        f1["data"].extend(f2["data"])
                        f1["configs"].update(help_configs)
                        f1["config_origins"].update(help_origns)



                        #f1['data'].extend(f2["data"][1:])
                        #f1['orgin'].update(f2["orgin"][1:])
                        with open(file_g, "w+") as f:
                            f1 = json.dumps(f1)
                            f.write(f1)                      
                        shutil.rmtree("{}/{}/{}/{}/del/".format(seed_path,method,runetime_folder, rf_seed))
                        shutil.rmtree("{}/{}/{}/{}/2t/del/".format(seed_path,method,runetime_folder, rf_seed))




