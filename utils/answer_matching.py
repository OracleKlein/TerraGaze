import re
import string
from typing import Any

from teonext.utils.bbox import extract_bboxes


RATIO_BINS = {
    "0%",
    "0%-10%",
    "10%-20%",
    "20%-30%",
    "30%-40%",
    "40%-50%",
    "50%-60%",
    "60%-70%",
    "70%-80%",
    "80%-90%",
    "90%-100%",
}
NEGATIVE_TEXTS = {
    "n a",
    "na",
    "no matching region",
    "none",
    "not applicable",
}
GRID_ALIASES = {
    "top left": "top-left",
    "top center": "top-center",
    "top centre": "top-center",
    "top right": "top-right",
    "middle left": "middle-left",
    "center left": "middle-left",
    "centre left": "middle-left",
    "central": "central",
    "center": "central",
    "centre": "central",
    "middle center": "central",
    "middle centre": "central",
    "center center": "central",
    "centre centre": "central",
    "middle right": "middle-right",
    "center right": "middle-right",
    "centre right": "middle-right",
    "lower left": "lower-left",
    "bottom left": "lower-left",
    "lower center": "lower-center",
    "lower centre": "lower-center",
    "bottom center": "lower-center",
    "bottom centre": "lower-center",
    "lower right": "lower-right",
    "bottom right": "lower-right",
}


def normalize_text(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("–", " ").replace("—", " ").replace("-", " ").replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_answer(text: Any) -> str:
    text = str(text).strip().lower()
    text = text.replace("&", " and ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def parse_candidate_classes(question: Any) -> list[str]:
    text = str(question or "")
    marker = "Choose from:"
    if marker not in text:
        return []
    tail = text.split(marker, 1)[1].strip()
    candidate_text = tail.split(" Output only", 1)[0].strip()
    if candidate_text.endswith("."):
        candidate_text = candidate_text[:-1].strip()
    return [part.strip() for part in candidate_text.split(",") if part.strip()]


def infer_answer_metadata(
    question: Any,
    ground_truth: Any = None,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    question_text = str(question or "")
    question_lower = question_text.lower()
    gt_text = str(ground_truth or "")
    answer_format = None

    if "comma-separated candidate class names" in question_lower:
        answer_format = "class_list"
    elif "[x_min, y_min, x_max, y_max]" in question_text or task == "spatial_referring_expression":
        answer_format = "bbox_xyxy_single"
    elif "only 'yes' or 'no'" in question_lower:
        answer_format = "yes_no"
    elif "ratio bin" in question_lower or canonical_ratio_bin(gt_text) is not None:
        answer_format = "ratio_bin"
    elif "grid-region" in question_lower or canonical_grid_region(gt_text) is not None:
        answer_format = "grid_region"
    elif "one image reference" in question_lower or canonical_image_label(gt_text) is not None:
        answer_format = "image_label"
    elif "image references" in question_lower or canonical_image_label_list(gt_text) is not None:
        answer_format = "image_label_list"
    elif "transition name" in question_lower:
        answer_format = "transition"
    elif "class name" in question_lower or "candidate class" in question_lower:
        answer_format = "class"
    elif "," in gt_text and task in {"question_answering", "region_based_question_answering"}:
        answer_format = "class_list"
    candidates = parse_candidate_classes(question_text)
    metadata: dict[str, Any] = {}
    if answer_format:
        metadata["answer_format"] = answer_format
    if candidates and answer_format in {"class", "class_list", "candidate_class", "transition"}:
        metadata["candidate_classes"] = candidates
    return metadata


def canonical_yes_no(text: Any) -> str | None:
    value = normalize_text(text)
    return value if value in {"yes", "no"} else None


def canonical_ratio_bin(text: Any) -> str | None:
    value = str(text or "").strip()
    match = re.fullmatch(r"(\d{1,3})\s*%\s*(?:-\s*(\d{1,3})\s*%)?", value)
    if not match:
        return None
    lower = int(match.group(1))
    upper = match.group(2)
    canonical = f"{lower}%" if upper is None else f"{lower}%-{int(upper)}%"
    return canonical if canonical in RATIO_BINS else None


def canonical_grid_region(text: Any) -> str | None:
    return GRID_ALIASES.get(normalize_text(text))


def canonical_image_label(text: Any) -> str | None:
    match = re.fullmatch(r"\s*image\s*([1-9]\d*)\s*\.?\s*", str(text or ""), flags=re.IGNORECASE)
    return f"Image {int(match.group(1))}" if match else None


def canonical_image_label_list(text: Any) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    labels: list[int] = []
    for part in raw.split(","):
        match = re.fullmatch(r"\s*image\s*([1-9]\d*)\s*\.?\s*", part, flags=re.IGNORECASE)
        if match is None:
            return None
        labels.append(int(match.group(1)))
    if not labels or len(set(labels)) != len(labels) or labels != sorted(labels):
        return None
    return ", ".join(f"Image {label}" for label in labels)


def is_negative_text(text: Any) -> bool:
    return normalize_text(text) in NEGATIVE_TEXTS


def canonical_label(text: Any) -> str:
    return normalize_text(text)


def _class_list_values(text: Any) -> list[str] | None:
    if normalize_text(text) == "none":
        return []
    parts = [part.strip() for part in str(text or "").split(",")]
    if not parts or any(not part for part in parts):
        return None
    return [canonical_label(part) for part in parts]


def _candidate_set(candidate_classes: Any) -> set[str]:
    if not candidate_classes:
        return set()
    return {canonical_label(candidate) for candidate in candidate_classes}


def _class_list_matches(response: Any, ground_truth: Any, candidate_classes: Any = None) -> bool:
    response_values = _class_list_values(response)
    gt_values = _class_list_values(ground_truth)
    if response_values is None or gt_values is None:
        return False
    candidates = _candidate_set(candidate_classes)
    if candidates and any(value not in candidates for value in response_values):
        return False
    return set(response_values) == set(gt_values)


def _bbox_text_matches(response: Any, ground_truth: Any, bbox_format: str = "bracket") -> bool:
    response_boxes = extract_bboxes(response, bbox_format=bbox_format)
    gt_boxes = extract_bboxes(ground_truth, bbox_format=bbox_format)
    return response_boxes == gt_boxes


def answer_matches(
    response: Any,
    ground_truth: Any,
    *,
    answer_format: str | None = None,
    candidate_classes: Any = None,
    question: Any = None,
    task: str | None = None,
    bbox_format: str = "bracket",
) -> bool:
    if answer_format is None:
        metadata = infer_answer_metadata(question, ground_truth, task=task)
        answer_format = metadata.get("answer_format")
        if candidate_classes is None:
            candidate_classes = metadata.get("candidate_classes")

    if answer_format in {"bbox_xyxy_single", "image_label", "image_label_list"} and is_negative_text(ground_truth):
        return is_negative_text(response)
    if answer_format == "class_list":
        return _class_list_matches(response, ground_truth, candidate_classes)
    if answer_format == "yes_no":
        response_value = canonical_yes_no(response)
        ground_truth_value = canonical_yes_no(ground_truth)
        return ground_truth_value is not None and response_value == ground_truth_value
    if answer_format == "ratio_bin":
        response_value = canonical_ratio_bin(response)
        ground_truth_value = canonical_ratio_bin(ground_truth)
        return ground_truth_value is not None and response_value == ground_truth_value
    if answer_format == "grid_region":
        response_value = canonical_grid_region(response)
        ground_truth_value = canonical_grid_region(ground_truth)
        return ground_truth_value is not None and response_value == ground_truth_value
    if answer_format == "image_label":
        response_value = canonical_image_label(response)
        ground_truth_value = canonical_image_label(ground_truth)
        return ground_truth_value is not None and response_value == ground_truth_value
    if answer_format == "image_label_list":
        response_value = canonical_image_label_list(response)
        ground_truth_value = canonical_image_label_list(ground_truth)
        return ground_truth_value is not None and response_value == ground_truth_value
    if answer_format in {"class", "candidate_class", "transition"}:
        return canonical_label(response) == canonical_label(ground_truth)
    if answer_format == "bbox_xyxy_single":
        return _bbox_text_matches(response, ground_truth, bbox_format=bbox_format)

    if task in {"spatial_referring_expression", "temporal_referring_expression"} and is_negative_text(ground_truth):
        return is_negative_text(response)
    return canonical_label(response) == canonical_label(ground_truth)


def annotate_answer_metadata(record: dict[str, Any]) -> dict[str, Any]:
    explicit_answer_format = record.get("answer_format")
    metadata = infer_answer_metadata(
        record.get("question"),
        record.get("ground_truth"),
        task=record.get("task"),
    )
    if explicit_answer_format is not None and explicit_answer_format != "":
        metadata.pop("answer_format", None)

    for key, value in metadata.items():
        if not value:
            continue
        existing = record.get(key)
        if key not in record or existing is None or existing == "":
            record[key] = value
    return record
