"""
play_config.py — Cấu hình nhanh cho HaxBall Play
==================================================
Chỉnh các thông số bên dưới rồi chạy:
    python play_config.py

Không cần gõ menu tương tác.
"""

from play import run, MAPS
import os

# ═══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH MATCH
# ═══════════════════════════════════════════════════════════════════════════════

# Chế độ: '1v1' | '2v2' | '3v3' | '4v4'
MODE = '1v1'

# Số người thật mỗi team (phần còn lại là bot)
#   RED tối đa 3 người (WASD+Space, Arrows+RCtrl, Numpad)
#   BLUE tối đa 2 người (TFGH+R, IJKL+U)
RED_HUMANS  = 1
BLUE_HUMANS = 0

# Số bàn thắng để kết thúc match (1–20)
WIN_SCORE = 5

# Giới hạn thời gian mỗi hiệp (phút, 1–6)
# Nếu hoà khi hết giờ → OVERTIME cho đến khi có bàn thắng
TIME_LIMIT_MIN = 3

# ═══════════════════════════════════════════════════════════════════════════════
#  MAP OVERRIDE (tuỳ chọn)
#  Để None → dùng map mặc định HaxMods theo MODE ở trên
#  Hoặc ghi đè từng giá trị:
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_MAP = None   # None = dùng preset; hoặc dict như bên dưới

# CUSTOM_MAP = dict(
#     HW     = 520.0,   # nửa chiều rộng sân (px)
#     HH     = 242.0,   # nửa chiều cao sân (px)
#     goal_y = 76.0,    # nửa chiều cao cửa gôn (px)
#     name   = 'My Custom Map',
# )

# ═══════════════════════════════════════════════════════════════════════════════
#  SPAWN MODE
#  'haxball' — vị trí spawn chuẩn HaxBall (khuyên dùng cho play)
#  'random'  — random mỗi nửa sân; bóng cách đều red/blue gần nhất
# ═══════════════════════════════════════════════════════════════════════════════
SPAWN_MODE = 'haxball'

# ═══════════════════════════════════════════════════════════════════════════════
#  AI MODEL PATH
#  Để None → dùng SimpleBot (chạy theo bóng)
#  Để đường dẫn tới file .pt → tải mạng PPO đã train
# ═══════════════════════════════════════════════════════════════════════════════
# Ví dụ: MODEL_PATH = 'training/experiment/1v1/checkpoints/best_model.pt'
MODEL_PATH = 'training\experiment\1v1\selfplay-snapshot_roundrobin\checkpoints\snapshot_000100.pt'

# ═══════════════════════════════════════════════════════════════════════════════
#  CONTROLS REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
# RED[0]  : W/A/S/D        + Space   (kick)
# RED[1]  : ↑↓←→           + RCtrl  (kick)
# RED[2]  : Numpad 8/2/4/6 + Num0   (kick)
# BLUE[0] : T/F/G/H        + R      (kick)   (T=up, F=left, G=down, H=right)
# BLUE[1] : I/J/K/L        + U      (kick)   (I=up, J=left, K=down, L=right)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  MAP PRESETS GỐC (tham khảo)
# ═══════════════════════════════════════════════════════════════════════════════
# MODE  | Map                  | HW    | HH    | goal_y
# ------+----------------------+-------+-------+-------
# 1v1   | Winky's Futsal       | 368   | 171   | 64
# 2v2   | Felon Network v2     | 520   | 242   | 76
# 3v3   | Futsal 3v3 Classic   | 630   | 270   | 80
# 4v4   | VALN v4              | 700   | 320   | 85


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    n = int(MODE[0])

    # Validate
    assert MODE in MAPS, f"MODE phải là {list(MAPS.keys())}"
    assert 0 <= RED_HUMANS  <= min(n, 3), f"RED_HUMANS phải trong [0, {min(n,3)}]"
    assert 0 <= BLUE_HUMANS <= min(n, 2), f"BLUE_HUMANS phải trong [0, {min(n,2)}]"
    assert 1 <= WIN_SCORE   <= 20,        "WIN_SCORE phải trong [1, 20]"
    assert 1 <= TIME_LIMIT_MIN <= 6,      "TIME_LIMIT_MIN phải trong [1, 6]"
    assert SPAWN_MODE in ('haxball', 'random')

    # Áp map override nếu có
    if CUSTOM_MAP is not None:
        MAPS[MODE] = CUSTOM_MAP

    print(f"\n{'='*52}")
    print(f"  HAXBALL  •  {MODE}  •  {MAPS[MODE]['name']}")
    print(f"{'='*52}")
    mp = MAPS[MODE]
    print(f"  Sân    : {int(mp['HW']*2)} × {int(mp['HH']*2)} px  |  Goal: ±{mp['goal_y']} px")
    print(f"  RED    : {RED_HUMANS} người + {n - RED_HUMANS} bot")
    print(f"  BLUE   : {BLUE_HUMANS} người + {n - BLUE_HUMANS} bot")
    print(f"  Thắng  : {WIN_SCORE} bàn  |  Giới hạn: {TIME_LIMIT_MIN} phút  |  Spawn: {SPAWN_MODE}")
    print(f"{'='*52}\n")

    run(
        mode           = MODE,
        n              = n,
        red_humans     = RED_HUMANS,
        blue_humans    = BLUE_HUMANS,
        win_score      = WIN_SCORE,
        time_limit_min = TIME_LIMIT_MIN,
        model_path     = MODEL_PATH,
    )
