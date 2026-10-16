from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /api/health

    Returns service readiness and model load status.
    Frontend should poll this before submitting predictions.

    Response 200:
    {
        "status":       "ok",
        "model_loaded": true,
        "num_classes":  13
    }
    """
    model_loaded = False
    num_classes  = 0

    try:
        from ..services.model_service import get_model, get_num_classes
        get_model()                    # triggers load if not yet done
        num_classes  = get_num_classes()
        model_loaded = True
    except Exception:
        pass

    leaf_gate_loaded = False
    try:
        from ..services.leaf_detector_service import load_leaf_detector
        load_leaf_detector()
        leaf_gate_loaded = True
    except Exception:
        pass

    return jsonify({
        "status":           "ok",
        "model_loaded":     model_loaded,
        "leaf_gate_loaded": leaf_gate_loaded,
        "num_classes":      num_classes,
    }), 200
