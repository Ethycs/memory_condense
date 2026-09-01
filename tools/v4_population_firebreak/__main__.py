"""CLI for the v4 evaluator-firebreak verifier."""

from __future__ import annotations

import argparse
import sys

from .analysis import (
    export_analysis_treatment_input,
    verify_analysis_treatment_input,
)
from .canonical import FirebreakError, canonical_json_bytes, publish_no_clobber
from .verifier import (
    export_confirmation_treatment_input,
    verify_evaluator_firebreak,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the exact v4 population and label-free treatment inputs"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--exposure-audit")
    parser.add_argument(
        "--treatment-input",
        action="append",
        help=(
            "supply once in analysis-only mode; otherwise repeat once for "
            "analysis and once for confirmation"
        ),
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="never decode confirmation histories/gold or open the exposure audit",
    )
    parser.add_argument(
        "--export-analysis-treatment",
        help="publish a no-clobber canonical analysis treatment artifact",
    )
    parser.add_argument(
        "--export-confirmation-treatment",
        help=(
            "publish the role-fixed, no-clobber canonical confirmation "
            "treatment artifact"
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        treatment_inputs = args.treatment_input or []
        if args.export_analysis_treatment and args.export_confirmation_treatment:
            raise FirebreakError("select exactly one treatment export mode")
        if args.export_confirmation_treatment:
            if args.analysis_only:
                raise FirebreakError(
                    "confirmation export cannot run in analysis-only mode"
                )
            if args.exposure_audit or treatment_inputs:
                raise FirebreakError(
                    "confirmation export cannot accept exposure or treatment inputs"
                )
            receipt = export_confirmation_treatment_input(
                dataset_path=args.dataset,
                split_manifest_path=args.split_manifest,
                output_path=args.export_confirmation_treatment,
            )
        elif args.export_analysis_treatment:
            if not args.analysis_only:
                raise FirebreakError(
                    "analysis export requires --analysis-only"
                )
            if args.exposure_audit or treatment_inputs:
                raise FirebreakError(
                    "analysis export cannot accept exposure or treatment inputs"
                )
            receipt = export_analysis_treatment_input(
                dataset_path=args.dataset,
                split_manifest_path=args.split_manifest,
                output_path=args.export_analysis_treatment,
            )
        elif args.analysis_only:
            if args.exposure_audit:
                raise FirebreakError(
                    "analysis-only verification cannot open an exposure audit"
                )
            if len(treatment_inputs) != 1:
                raise FirebreakError(
                    "analysis-only verification requires exactly one treatment input"
                )
            receipt = verify_analysis_treatment_input(
                dataset_path=args.dataset,
                split_manifest_path=args.split_manifest,
                treatment_input_path=treatment_inputs[0],
            )
        else:
            if not args.exposure_audit:
                raise FirebreakError(
                    "confirmation lock mode requires --exposure-audit"
                )
            if len(treatment_inputs) != 2:
                raise FirebreakError(
                    "confirmation lock mode requires exactly two treatment inputs"
                )
            receipt = verify_evaluator_firebreak(
                dataset_path=args.dataset,
                split_manifest_path=args.split_manifest,
                exposure_audit_path=args.exposure_audit,
                treatment_input_paths=treatment_inputs,
            )
        payload = canonical_json_bytes(receipt) + b"\n"
        if args.output:
            publish_no_clobber(args.output, payload)
        else:
            sys.stdout.buffer.write(payload)
        return 0
    except FirebreakError as exc:
        print(f"firebreak verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
