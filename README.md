# V5.27.9 – HYBRID SMART PDF

Bản này tiếp tục cơ chế kết hợp V5.25.3 + V5.26.9 theo từng thành phần của từng câu.

Điểm sửa chính:
- Phần II nhận đúng nhãn chữ thường `a) b) c) d)`; nếu lớp text PDF làm mất dấu `)`/`.` và chỉ còn `a ...`, `c ...`, `d ...` thì vẫn tách đúng.
- Quy tắc nhãn không dấu câu chỉ áp dụng cho câu Đúng/Sai, không áp dụng cho Phần I.
- Giữ sửa V5.27.7: không tự sinh Câu 13 khi PDF chuyển từ Phần I sang Phần II.
- PDF sạch vẫn ưu tiên semantic/LaTeX; thành phần không an toàn mới fallback sang visual.

Xem `README_V5.27.9_FIX_PHAN_II_CHU_THUONG.txt` để biết chi tiết.
