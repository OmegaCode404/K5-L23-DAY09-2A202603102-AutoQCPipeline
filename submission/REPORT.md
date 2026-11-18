 # Day 09 Lab  — 2A202603102 -Vũ Minh Châu

## 1. Kết luận trước
 pipeline kết luận Hold. Pipeline quét 224 ảnh vào, 205 ảnh ra dataset. 

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh / số dòng ghi nhận | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_CLASS_ID | 6 ảnh / 6 dòng | | |
| BAD_CLASS_NAME | 1 ảnh / 2 dòng | | |
| BAD_FORMAT | 3 ảnh / 3 dòng | | |
| BAD_SIZE | 3 ảnh / 3 dòng | | |
| BOX_HUGE | 3 ảnh / 3 dòng | | |
| BOX_OUT_OF_IMAGE | 8 ảnh / 8 dòng | | |
| BOX_TINY | 4 ảnh / 6 dòng | | |
| BOX_ZERO | 4 ảnh / 4 dòng | | |
| CLASS_RARE | 1 ảnh / 4 dòng | | |
| CORRUPT_IMAGE | 2 ảnh / 2 dòng | | |
| DUP_BOX | 7 ảnh / 7 dòng | | |
| DUP_IMAGE_EXACT | 8 ảnh / 8 dòng | | |
| DUP_IMAGE_NEAR | 7 ảnh / 7 dòng | | |
| EMPTY_LABEL | 11 ảnh / 11 dòng | | |
| LEAK_TEST | 11 ảnh / 11 dòng | | |
| MISSING_LABEL_FILE | 6 ảnh / 6 dòng | | |
| POSSIBLE_FALSE_POS | 26 ảnh / 28 dòng | | |
| POSSIBLE_LOOSE_BOX | 23 ảnh / 27 dòng | | |
| POSSIBLE_MISSING | 52 ảnh / 53 dòng | | |
| POSSIBLE_WRONG_CLASS | 17 ảnh / 17 dòng | | |


`05_review_queue.csv` xếp ảnh đầu hàng theo điểm ưu tiên cao nhất
Với ngân sách 80 ảnh, người xem được 185 lỗi ngữ nghĩa


## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 |`DUP_IMAGE_EXACT`: 1.00/1.00 → 1.00/1.00; `DUP_IMAGE_NEAR`: 1.00/0.50 → 1.00/0.83 |recall tăng, precision trên calibration vẫn 1.00. |
| `second_opinion.conf` | 0.25 | 0.6 |`WRONG_CLASS`: 0.56/0.83 → 0.63/0.83; `MISSING`: 0.50/1.00 → 0.57/1.00; `LOOSE_BOX`: 0.29/0.40 → 0.20/0.20; `FALSE_POS`: 0.31/1.00 → 0.31/1.00 |precision của một số cảnh báo tăng nhưng bắt lỗi box lỏng kém hơn; false-positive không đổi |

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
 (a) với ngân sách duyệt bằng 0, không có nhãn nào được người duyệt sửa; lỗi bắt buộc còn lại khiến gate chọn Rework 
 (b) ≥ 2 điểm mù có bằng chứng (box lỏng nhẹ, nhầm lớp mà model lớn cũng nhầm, ảnh trùng biến đổi mạnh…).

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
chưa chạy

## 6. Quyết định và ghi chú AI
 (a) Để thành Release, cần không còn lỗi bắt buộc và giảm cảnh báo chưa duyệt xuống tối đa 20/205 ảnh (dưới 10%)
 (b) Dùng AI viết code? Ghi 3 dòng: prompt chính, cách bạn kiểm code AI, 1 lần AI sai. Không dùng: ghi "không dùng".

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
