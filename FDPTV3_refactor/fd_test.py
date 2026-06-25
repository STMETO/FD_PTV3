"""FDPTV3_refactor 联邦测试入口。"""

from pointcept.engines.defaults import default_config_parser
from pointcept.engines.launch import launch

from .evaluation.tester import build_argument_parser, main_worker


def main():
    args = build_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg.weight = args.weight
    cfg.save_path = args.save_path

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
