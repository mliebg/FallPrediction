import argparse
import subprocess
import datetime
import json
#import os

def get_config_file():
    with open("fallpred.config.json", 'r') as file:
        config = json.load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="CLI for NAO FallPrediction")
    subparsers = parser.add_subparsers(dest="command")

    # data command - Creating training data
    parser_script1 = subparsers.add_parser("data", help="Create training data for a model")
    # Add Arguments

    # build command - Build & train a model
    parser_script2 = subparsers.add_parser("build", help="Build and train a ML model")
    # Add arguments

    # eval command - Evaluate a model
    parser_script3 = subparsers.add_parser("eval", help="Calculate and visualize evaluation of a trained model")

    args = parser.parse_args()

    logfile_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + f".{args.command}.log"

    if args.command == "data":
        #script_path = os.path.join("script1_dir", "script1.py")
        subprocess.run(["python3", "createModelDataSmallTs.py", logfile_name])
    elif args.command == "build":
        subprocess.run(["python3", "lstmBuildNTrain.py", logfile_name])
    elif args.command == "eval":
        subprocess.run(["python3", "testProcessLogData.py", logfile_name])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
