"""Local Flask application for blinded IAD spectrogram localization."""

from __future__ import annotations

from io import BytesIO
import math
import os
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, abort, jsonify, make_response, render_template, request, send_file
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from iad_labeling.schema import (
    ALLOWED_STATUSES,
    AnnotationValidationError,
    SUGGESTED_AMBIGUITY_REASONS,
    validate_annotator,
)
from iad_labeling.store import (
    AnnotationStore,
    ManifestValidationError,
    PilotManifest,
    ProposalStore,
    ProposalValidationError,
    StoragePathError,
    render_uint8,
)


def _finite_range(
    minimum: float | None, maximum: float | None
) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        raise ValueError("global_min and global_max must be provided together")
    lower, upper = float(minimum), float(maximum)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("global contrast values must be finite and increasing")
    return lower, upper


def create_app(
    *,
    manifest_path: str | os.PathLike[str],
    proposals_dir: str | os.PathLike[str] | None,
    annotations_dir: str | os.PathLike[str],
    annotator_id: str,
    source_root: str | os.PathLike[str] | None = None,
    global_min: float | None = None,
    global_max: float | None = None,
    testing: bool = False,
) -> Flask:
    """Create the annotation app without binding a network interface."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.config.update(
        MAX_CONTENT_LENGTH=256 * 1024,
        JSON_SORT_KEYS=False,
        TESTING=testing,
    )

    manifest = PilotManifest(manifest_path, source_root=source_root)
    proposals = ProposalStore(proposals_dir)
    annotations = AnnotationStore(annotations_dir)
    configured_annotator = validate_annotator(annotator_id)
    configured_global_range = _finite_range(global_min, global_max)

    app.extensions["iad_manifest"] = manifest
    app.extensions["iad_proposals"] = proposals
    app.extensions["iad_annotations"] = annotations
    app.extensions["iad_annotator"] = configured_annotator

    def sample_or_404(sample_id: str):
        try:
            return manifest.get(sample_id)
        except KeyError:
            abort(404, description="unknown or invalid sample_id")

    def proposal_or_error(sample_id: str) -> dict[str, Any] | None:
        try:
            return proposals.load(sample_id)
        except ProposalValidationError as exc:
            abort(422, description=f"invalid proposal artifact: {exc}")

    def global_range() -> tuple[float, float]:
        return configured_global_range or manifest.global_contrast()

    def enforce_configured_annotator(value: Any, *, required: bool = False) -> str:
        if value is None or value == "":
            if required:
                abort(400, description="annotator is required")
            return configured_annotator
        supplied = validate_annotator(value)
        if supplied != configured_annotator:
            abort(
                403,
                description=(
                    "annotator does not match the reviewer identity locked "
                    "for this server"
                ),
            )
        return configured_annotator

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_error):
        return jsonify(error="request payload is too large"), 413

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(error=error.description), error.code
        return error

    def validation_error(error):
        return jsonify(error=str(error)), 400

    for exception_type in (
        AnnotationValidationError,
        ManifestValidationError,
        ProposalValidationError,
        StoragePathError,
        ValueError,
    ):
        app.register_error_handler(exception_type, validation_error)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            sample_count=len(manifest),
            statuses=sorted(ALLOWED_STATUSES),
            ambiguity_reasons=SUGGESTED_AMBIGUITY_REASONS,
            annotator_id=configured_annotator,
        )

    @app.get("/api/samples")
    def list_samples():
        annotator = enforce_configured_annotator(request.args.get("annotator"))
        summaries: Mapping[str, Mapping[str, Any]] = annotations.summaries(annotator)
        samples = []
        for record in manifest.records:
            item = record.public_dict()
            item["proposal_available"] = proposals.exists(record.sample_id)
            item["annotation"] = summaries.get(record.sample_id)
            samples.append(item)
        return jsonify(samples=samples, count=len(samples))

    @app.get("/api/samples/<sample_id>")
    def sample_detail(sample_id: str):
        record = sample_or_404(sample_id)
        item = record.public_dict()
        item["proposal_available"] = proposals.exists(record.sample_id)
        annotator = enforce_configured_annotator(request.args.get("annotator"))
        item["annotation"] = annotations.load(record.sample_id, annotator)
        return jsonify(sample=item)

    @app.get("/api/samples/<sample_id>/image")
    def sample_image(sample_id: str):
        record = sample_or_404(sample_id)
        mode = request.args.get("contrast", "global")
        if mode not in {"global", "percentile"}:
            abort(400, description="contrast must be 'global' or 'percentile'")
        sample = manifest.load_array(record.sample_id)
        image_array, used_range = render_uint8(
            sample,
            mode=mode,
            global_range=global_range(),
        )
        buffer = BytesIO()
        Image.fromarray(image_array, mode="L").save(buffer, format="PNG")
        buffer.seek(0)
        response = make_response(
            send_file(
                buffer,
                mimetype="image/png",
                download_name=f"{record.sample_id}.png",
                max_age=0,
            )
        )
        response.headers["X-IAD-Image-Shape"] = "64x64"
        response.headers["X-IAD-Contrast-Mode"] = mode
        response.headers["X-IAD-Contrast-Lower"] = f"{used_range[0]:.9g}"
        response.headers["X-IAD-Contrast-Upper"] = f"{used_range[1]:.9g}"
        return response

    @app.get("/api/samples/<sample_id>/proposal")
    def sample_proposal(sample_id: str):
        record = sample_or_404(sample_id)
        proposal = proposal_or_error(record.sample_id)
        if proposal is None:
            abort(404, description="proposal is unavailable for this sample")
        # proposal contains a binary mask and bbox only. score_map values are
        # deliberately absent from every HTTP response.
        return jsonify(proposal=proposal)

    @app.get("/api/samples/<sample_id>/annotation")
    def get_annotation(sample_id: str):
        record = sample_or_404(sample_id)
        annotator = enforce_configured_annotator(request.args.get("annotator"))
        annotation = annotations.load(record.sample_id, annotator)
        return jsonify(annotation=annotation)

    @app.put("/api/samples/<sample_id>/annotation")
    def put_annotation(sample_id: str):
        record = sample_or_404(sample_id)
        payload = request.get_json(silent=False)
        if not isinstance(payload, Mapping):
            abort(400, description="request body must be a JSON object")
        annotator = enforce_configured_annotator(
            payload.get("annotator"),
            required=True,
        )
        # A direct PUT must satisfy the same source identity check as image
        # rendering; do not rely on the browser having fetched the image first.
        manifest.load_array(record.sample_id)
        existed = annotations.load(record.sample_id, annotator) is not None
        proposal = proposal_or_error(record.sample_id)
        annotation = annotations.save(
            sample_id=record.sample_id,
            sample_hash=record.sample_hash,
            sample_label=record.label,
            annotator=annotator,
            payload=payload,
            proposal_provenance=None if proposal is None else proposal["provenance"],
        )
        return jsonify(annotation=annotation), (200 if existed else 201)

    return app
