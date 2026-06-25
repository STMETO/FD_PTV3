"""FDPTV3_refactor 联邦训练入口。"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)

from .server.builder import build_server


def main_worker(cfg):
    server = build_server(cfg)
    server.run()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
