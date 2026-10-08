"""Command-line entry point: run (or plan) an experiment from its config file.

    python -m elements.run configs/supervised_ladder.yaml --plan
    python -m elements.run configs/supervised_ladder.yaml
    python -m elements.run configs/novelty_ladder.yaml --set seed=7 --set device=cpu

``--plan`` prints the steps, head sizes, memory and a rough time estimate and
exits without training. ``--set key=value`` overrides any field, dotted for
nested ones (``--set data.path=D:/data/nsl.csv``); values are parsed as YAML,
so numbers and lists work (``--set arms=[quantum,dials]``).
"""

from __future__ import annotations

import argparse

import yaml

from elements import config as config_mod
from elements.experiments.runner import plan, run


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="YAML or JSON experiment config")
    p.add_argument("--plan", action="store_true", help="print the plan and exit")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config field (repeatable)")
    args = p.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    cfg = config_mod.load(args.config, overrides)

    print(plan(cfg))
    if not args.plan:
        print()
        run(cfg)


if __name__ == "__main__":
    main()
