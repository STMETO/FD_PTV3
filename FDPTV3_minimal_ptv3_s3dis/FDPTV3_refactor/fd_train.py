"""Minimal federated training entrypoint.

The minimal project intentionally exposes only one canonical training command:
``python -m FDPTV3_refactor.fd_train``. This module is therefore the first file
to inspect when tracing the execution chain.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)

from .server.builder import build_server


def main_worker(cfg):
    # The server object owns the whole federated lifecycle: environment setup,
    # global model construction, round orchestration, validation and final test.
    server = build_server(cfg)
    server.run()


def main():
    # Reuse Pointcept's standard argument parser so the extracted project keeps
    # the same CLI surface as the original repository.
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
