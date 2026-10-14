"""Command line entry point."""
from __future__ import annotations

import argparse

from .config import dump_config, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kendo-vision",
                                     description="Strike detection and ippon judging for kendo video.")
    parser.add_argument("--config", help="YAML overriding the defaults (configs/default.yaml is used if present)")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="analyse a video file or a live camera")
    run.add_argument("--source", required=True, help="video path, or a camera index such as 0")
    run.add_argument("--red-side", choices=["left", "right"], help="where red stands at the start")
    run.add_argument("--save", help="write the annotated video here")
    run.add_argument("--events", help="write judged strikes here as JSON lines")
    run.add_argument("--dump-features", help="write per-frame features here (.npz) for train-strike")
    run.add_argument("--show", action="store_true", help="show the annotated video while running")
    run.add_argument("--no-audio", action="store_true")

    extract = commands.add_parser("extract-frames", help="save frames from a video for labelling")
    extract.add_argument("--video", required=True)
    extract.add_argument("--out", required=True, help="e.g. data/shinai/images/train")
    extract.add_argument("--every", type=float, default=0.5, help="seconds between saved frames")
    extract.add_argument("--events", help="also save every frame around these events")
    extract.add_argument("--pad", type=float, default=0.3, help="seconds either side of each event")

    label_shinai = commands.add_parser("label-shinai", help="label shinai keypoints on extracted frames")
    label_shinai.add_argument("--images", required=True)
    label_shinai.add_argument("--no-prefill", action="store_true", help="start each image empty")

    label_events = commands.add_parser("label-events", help="confirm, fix or add strike events on a video")
    label_events.add_argument("--video", required=True)
    label_events.add_argument("--out", required=True, help="label file to write (and resume from)")
    label_events.add_argument("--events", help="candidates from `run --events` to start from")

    train_shinai = commands.add_parser("train-shinai", help="train the shinai keypoint model")
    train_shinai.add_argument("--data", default="data/shinai")
    train_shinai.add_argument("--base", help="weights to start from (default: the pose weights)")
    train_shinai.add_argument("--epochs", type=int, default=150)
    train_shinai.add_argument("--image-size", type=int, default=1280)
    train_shinai.add_argument("--batch", type=int, default=8)

    train_strike = commands.add_parser("train-strike", help="train StrikeNet on labelled recordings")
    train_strike.add_argument("--data", default="data/strikes", help="folder of <name>.npz + <name>.jsonl pairs")
    train_strike.add_argument("--val", default="", help="comma-separated recording names to hold out")
    train_strike.add_argument("--epochs", type=int, default=60)

    evaluate = commands.add_parser("eval-events", help="score predicted events against labels")
    evaluate.add_argument("--pred", required=True)
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--tolerance", type=float, default=0.25,
                          help="seconds two events may differ and still match")

    commands.add_parser("config", help="print the active configuration as YAML")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _build_parser().parse_args(argv)
    config = load_config(arguments.config)

    # each command imports what it needs only when chosen, so --help stays fast
    if arguments.command == "run":
        from .pipeline import run

        if arguments.red_side:
            config.track.red_side = arguments.red_side
        if arguments.no_audio:
            config.audio.enabled = False
        run(config, arguments.source, arguments.save, arguments.events, arguments.dump_features, arguments.show)
    elif arguments.command == "extract-frames":
        from .tools.extract_frames import extract

        extract(arguments.video, arguments.out, arguments.every, arguments.events, arguments.pad)
    elif arguments.command == "label-shinai":
        from .tools.label_shinai import run

        run(arguments.images, config, prefill=not arguments.no_prefill)
    elif arguments.command == "label-events":
        from .tools.label_events import run

        run(arguments.video, arguments.out, arguments.events)
    elif arguments.command == "train-shinai":
        from .training.train_shinai import train

        train(arguments.data, arguments.base or config.model.pose_weights, arguments.epochs,
              arguments.image_size, arguments.batch, config.model.shinai_weights)
    elif arguments.command == "train-strike":
        from .training.train_strike import train

        held_out = [name for name in arguments.val.split(",") if name]
        train(arguments.data, held_out, config, arguments.epochs)
    elif arguments.command == "eval-events":
        from .training.eval_events import evaluate

        evaluate(arguments.pred, arguments.truth, arguments.tolerance)
    elif arguments.command == "config":
        print(dump_config(config), end="")


if __name__ == "__main__":
    main()
