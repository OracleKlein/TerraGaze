import re
from copy import deepcopy


BBOX_TEXT_FORMAT_BRACKET = "bracket"
BBOX_TEXT_FORMAT_COORD_TOKEN = "coord_token"
BBOX_COORD_TOKEN_MAX = 1000
BBOX_COORD_TOKENS = tuple(f"<{index}>" for index in range(BBOX_COORD_TOKEN_MAX + 1))
BBOX_TEXT_FORMATS = {
    BBOX_TEXT_FORMAT_BRACKET,
    BBOX_TEXT_FORMAT_COORD_TOKEN,
}

BRACKET_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
COORD_TOKEN_BBOX_RE = re.compile(r"<(\d+)><(\d+)><(\d+)><(\d+)>")


def normalize_bbox_text_format(bbox_format):
    if isinstance(bbox_format, bool):
        return BBOX_TEXT_FORMAT_COORD_TOKEN if bbox_format else BBOX_TEXT_FORMAT_BRACKET
    if bbox_format is None:
        return BBOX_TEXT_FORMAT_BRACKET
    bbox_format = str(bbox_format)
    if bbox_format not in BBOX_TEXT_FORMATS:
        supported = ", ".join(sorted(BBOX_TEXT_FORMATS))
        raise ValueError(f"Unsupported bbox text format: {bbox_format!r}. Supported values: {supported}.")
    return bbox_format


def bbox_text_format_from_config(config):
    return (
        BBOX_TEXT_FORMAT_COORD_TOKEN
        if bool(getattr(config, "use_bbox_coord_tokens", False))
        else BBOX_TEXT_FORMAT_BRACKET
    )


def bbox_coord_tokens(max_coord=BBOX_COORD_TOKEN_MAX):
    max_coord = int(max_coord)
    if max_coord < 0:
        raise ValueError(f"bbox_coord_token_max must be >= 0, got {max_coord}.")
    return [f"<{index}>" for index in range(max_coord + 1)]


def _format_sample_hint(sample_hint):
    return f" ({sample_hint})" if sample_hint else ""


def _validate_bbox_coordinates(values, *, max_coord=BBOX_COORD_TOKEN_MAX, sample_hint=None):
    if len(values) != 4:
        raise ValueError(f"BBox must contain exactly 4 coordinates{_format_sample_hint(sample_hint)}: {values!r}")

    integers = []
    for raw_value in values:
        value = float(raw_value)
        if not value.is_integer():
            raise ValueError(
                f"BBox coordinates must be integers in coord-token mode"
                f"{_format_sample_hint(sample_hint)}: {values!r}"
            )
        int_value = int(value)
        if int_value < 0 or int_value > max_coord:
            raise ValueError(
                f"BBox coordinate {int_value} is outside [0, {max_coord}]"
                f"{_format_sample_hint(sample_hint)}: {values!r}"
            )
        integers.append(int_value)

    x1, y1, x2, y2 = integers
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"BBox must be valid xyxy coordinates with x2 > x1 and y2 > y1"
            f"{_format_sample_hint(sample_hint)}: {values!r}"
        )
    return tuple(integers)


def convert_bracket_bboxes_to_coord_tokens(text, *, max_coord=BBOX_COORD_TOKEN_MAX, sample_hint=None):
    text = str(text)

    def replace(match):
        coords = _validate_bbox_coordinates(
            match.groups(),
            max_coord=max_coord,
            sample_hint=sample_hint,
        )
        return "".join(f"<{coord}>" for coord in coords)

    return BRACKET_BBOX_RE.sub(replace, text)


def convert_conversation_bboxes_to_coord_tokens(conversations, *, max_coord=BBOX_COORD_TOKEN_MAX, sample_hint=None):
    converted = deepcopy(conversations)
    for turn_index, turn in enumerate(converted):
        if not isinstance(turn, dict):
            continue
        for key in ("value", "content"):
            if key not in turn:
                continue
            turn_hint = f"{sample_hint}, turn={turn_index}" if sample_hint else f"turn={turn_index}"
            turn[key] = convert_bracket_bboxes_to_coord_tokens(
                turn[key],
                max_coord=max_coord,
                sample_hint=turn_hint,
            )
    return converted


def extract_bboxes(text, *, bbox_format=BBOX_TEXT_FORMAT_BRACKET, max_coord=BBOX_COORD_TOKEN_MAX, validate=False):
    bbox_format = normalize_bbox_text_format(bbox_format)
    text = str(text)
    if bbox_format == BBOX_TEXT_FORMAT_BRACKET:
        boxes = [tuple(float(value) for value in match.groups()) for match in BRACKET_BBOX_RE.finditer(text)]
    else:
        boxes = [tuple(float(value) for value in match.groups()) for match in COORD_TOKEN_BBOX_RE.finditer(text)]

    if validate:
        return [
            tuple(
                float(value)
                for value in _validate_bbox_coordinates(box, max_coord=max_coord)
            )
            for box in boxes
        ]
    return boxes


def count_bbox_candidates(text, *, bbox_format=BBOX_TEXT_FORMAT_BRACKET):
    bbox_format = normalize_bbox_text_format(bbox_format)
    text = str(text)
    if bbox_format == BBOX_TEXT_FORMAT_COORD_TOKEN:
        return len(COORD_TOKEN_BBOX_RE.findall(text)) + len(BRACKET_BBOX_RE.findall(text))
    return len(BRACKET_BBOX_RE.findall(text))


def contains_bbox(text, *, bbox_format=BBOX_TEXT_FORMAT_BRACKET):
    return count_bbox_candidates(text, bbox_format=bbox_format) > 0


def normalize_bbox_scale(box, coordinate_format="auto"):
    if coordinate_format == "thousandth":
        denominator = 1000.0
    elif coordinate_format == "percent":
        denominator = 100.0
    elif coordinate_format == "auto":
        denominator = 1000.0 if max(box) > 100.0 else 100.0
    else:
        raise ValueError(f"Unsupported bbox coordinate format: {coordinate_format}")
    return tuple(value / denominator for value in box)


def bbox_area_normalized(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_area_normalized(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def bbox_iou_normalized(box_a, box_b):
    intersection = bbox_intersection_area_normalized(box_a, box_b)
    union = bbox_area_normalized(box_a) + bbox_area_normalized(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center_normalized(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_center_error_normalized(box_a, box_b):
    ax, ay = bbox_center_normalized(box_a)
    bx, by = bbox_center_normalized(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def bbox_coverage_recall_normalized(pred_boxes, gt_box):
    gt_area = bbox_area_normalized(gt_box)
    if gt_area <= 0:
        return 0.0
    return max(
        (
            bbox_intersection_area_normalized(pred_box, gt_box) / gt_area
            for pred_box in pred_boxes
        ),
        default=0.0,
    )
