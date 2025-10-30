import argparse
import sys
import os
from os.path import join, dirname
from datetime import datetime as dt

from loguru import logger
import torch

from src.utils.configs import load_args, dump_args
from src.utils.general import seed_torch
from src.engine import OAKTrainer


def main():
    """
    Command line arguments for two purposes: (1) key args, and (2) override config file
    """
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument("config", type=str, help="Config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--eval_path", type=str, default="", help="Path to evaluation data")
    # override config file
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    cli_args = parser.parse_args()

    """
    Prepare specific configs and initialize experiments
    """
    args = load_args(cli_args.config, cli_opts=cli_args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.SEED = int(cli_args.seed)
    if args.SEED >= 0:
        seed_torch(args.SEED)
        args.EXP_NAME = args.EXP_NAME + "_SEED{}".format(str(args.SEED))

    args.DEBUG = cli_args.debug
    if not args.DEBUG:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    
    if cli_args.eval_path:
        args.OUTPUT_DIR = dirname(cli_args.eval_path)
        args.logger = logger
        trainer = OAKTrainer(args, device)

        ckpt = torch.load(cli_args.eval_path, map_location="cpu")
        msg = trainer.model.load_state_dict(ckpt, strict=True)
        args.logger.info('Evaluation only: loaded model from {}'.format(cli_args.eval_path))
        args.logger.info(msg)

        """
        Start evaluation
        """
        trainer.evaluate(override_unlabeled_class_names=None, logger=logger)

    else:
        args.OUTPUT_DIR = join(args.SAVE_DIR, "{}_{}".format(args.EXP_NAME, dt.now().isoformat()))
        os.makedirs(args.OUTPUT_DIR, exist_ok=False)
        dump_args(args, join(args.OUTPUT_DIR, "config.yaml"))
        args.logger = logger
        
        """
        Start training
        """
        trainer = OAKTrainer(args, device)
        trainer.train()


if __name__ == "__main__":
    main()