"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    issues_by_image = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            issues_by_image[issue.image].append(issue)

    fixes = []
    image_paths = defaultdict(list)
    for path in ds.image_files():
        image_paths[path.stem].append(path)

    removed_stems = set()
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    for stem, image_issues in issues_by_image.items():
        should_remove = any(
            issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes
            for issue in image_issues
        )
        if not should_remove:
            continue

        removed_stems.add(stem)
        paths = image_paths.get(stem, [])
        if not paths:
            continue

        label_path = ds.label_path(stem)
        before = [path.name for path in paths]
        for path in paths:
            path.unlink()
        if label_path.exists():
            before.append(label_path.name)
            label_path.unlink()

        fixes.append({
            "image": stem,
            "line": None,
            "action": "remove_image",
            "before": ", ".join(before),
            "after": "(đã xoá)",
        })

    if any(issue.code == "BAD_CLASS_NAME" for issue in issues_by_image.get("data.yaml", [])):
        expected_names = dict(enumerate(cfg["classes"]))
        if ds.yaml_names() != expected_names:
            yaml_path = ds.yaml_path
            before = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
            data = yaml.safe_load(before)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise ValueError("data.yaml phải chứa một mapping ở cấp cao nhất")
            data["names"] = expected_names
            after = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
            with yaml_path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(after)
            fixes.append({
                "image": "data.yaml",
                "line": None,
                "action": "rename_classes",
                "before": before,
                "after": after,
            })

    for stem, image_issues in issues_by_image.items():
        if stem == "data.yaml" or stem in removed_stems:
            continue

        path = ds.label_path(stem)
        has_missing_label = any(issue.code == "MISSING_LABEL_FILE" for issue in image_issues)
        if has_missing_label and image_paths.get(stem) and not path.exists():
            fixes.append({
                "image": stem,
                "line": None,
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

        line_codes = defaultdict(set)
        for issue in image_issues:
            if issue.line is not None:
                line_codes[issue.line].add(issue.code)
        if not line_codes or not path.exists():
            continue

        content = path.read_text(encoding="utf-8")
        lines: list[str | None] = content.splitlines(keepends=True)
        changed = False

        for line_no in sorted(line_codes):
            if not isinstance(line_no, int):
                continue
            index = line_no - 1
            if index < 0 or index >= len(lines):
                continue

            original = lines[index]
            if original.endswith("\r\n"):
                body, ending = original[:-2], "\r\n"
            elif original.endswith(("\n", "\r")):
                body, ending = original[:-1], original[-1]
            else:
                body, ending = original, ""

            codes = line_codes[line_no]
            if not body.strip():
                continue
            if codes.intersection({"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}):
                lines[index] = None
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": body,
                    "after": "(đã xoá)",
                })
                changed = True
                continue

            if "BOX_OUT_OF_IMAGE" not in codes:
                continue

            class_id, cx, cy, width, height = parse_yolo_line(body)
            left = max(0.0, cx - width / 2)
            top = max(0.0, cy - height / 2)
            right = min(1.0, cx + width / 2)
            bottom = min(1.0, cy + height / 2)
            if right <= left or bottom <= top:
                lines[index] = None
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": body,
                    "after": "(đã xoá)",
                })
                changed = True
                continue

            clipped = (
                (left + right) / 2,
                (top + bottom) / 2,
                right - left,
                bottom - top,
            )
            after = f"{class_id} " + " ".join(repr(value) for value in clipped)
            if after != body:
                lines[index] = after + ending
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "clip_box",
                    "before": body,
                    "after": after,
                })
                changed = True

        if changed:
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write("".join(line for line in lines if line is not None))

    return fixes
