"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    settings = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []

    for image in ds.image_files():
        stem = image.stem
        if stem not in preds:
            continue
        if ds.label_path(stem).is_file() is False:
            continue

        labels = ds.valid_boxes(stem)
        if not labels:
            labels = []

        size = ds.image_size(stem)
        primary_preds = []
        fp_preds = []

        for row in preds.get(stem, []):
            try:
                class_id, conf, cx, cy, width, height = row
                class_id = int(class_id)
                conf = float(conf)
                box = (float(cx), float(cy), float(width), float(height))
            except (TypeError, ValueError):
                continue

            if conf >= settings["fp_conf"]:
                fp_preds.append((box, conf))

            if conf < settings["conf"]:
                continue
            if size is None:
                continue

            image_width, image_height = size
            area_px = box[2] * image_width * box[3] * image_height
            if area_px >= settings["min_area_px"]:
                primary_preds.append((class_id, box, conf))

        for class_id, box, conf in primary_preds:
            candidates = [
                (label_class, (label_cx, label_cy, label_w, label_h))
                for label_class, label_cx, label_cy, label_w, label_h in labels
            ]
            cands_with_iou = [
                (label_class, label_box, iou_xywh(box, label_box))
                for label_class, label_box in candidates
            ]
            cands = [
                (label_class, label_box, iou)
                for label_class, label_box, iou in cands_with_iou
                if iou >= settings["iou_missing"]
            ]

            if not cands:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_MISSING", stem,
                    "model lớn hơn thấy vật có thể chưa có nhãn",
                    extra={"conf": conf},
                ))
                continue

            same_class = [
                item for item in cands
                if item[0] == class_id
            ]
            if same_class:
                best_iou = max(item[2] for item in same_class)
                if best_iou < settings["iou_loose"]:
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                        "nhãn cùng lớp có thể bị lệch hoặc box quá lỏng",
                        extra={"iou": best_iou},
                    ))
                continue

            other_class = [
                item for item in cands
                if item[0] != class_id
            ]
            if other_class:
                best_iou = max(item[2] for item in other_class)
                if best_iou >= settings["iou_match"]:
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                        "vị trí trùng vật có thể bị gán sai lớp",
                        extra={"iou": best_iou},
                    ))

        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                label = parse_yolo_line(text)
            except ValueError:
                continue

            label_class, cx, cy, width, height = label
            if not 0 <= label_class < len(ds.classes):
                continue

            label_box = (cx, cy, width, height)
            max_iou = 0.0
            matched = False
            for pred_box, _ in fp_preds:
                iou = iou_xywh(label_box, pred_box)
                max_iou = max(max_iou, iou)
                if iou >= settings["iou_missing"]:
                    matched = True
                    break

            if not matched:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                    "box nhãn có thể thừa so với dự đoán của model lớn hơn",
                    line=line_no,
                    extra={"iou": max_iou},
                ))

    return issues
