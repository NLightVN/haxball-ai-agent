"""
play.py — HaxBall Interactive Play
===================================
Chạy game HaxBall với người thật + bot, renderer bằng Pygame.

Sử dụng:
    python play.py                           # menu tương tác
    python play.py --mode 2v2 --red 1 --blue 1

Controls (tối đa 3 người/team):
    RED[0]  : W/A/S/D + Space (kick)
    RED[1]  : Arrow Up/Down/Left/Right + RCtrl (kick)
    RED[2]  : Numpad 8/2/4/6 + Numpad 0 (kick)
    BLUE[0] : T/G/F/H + R (kick)
    BLUE[1] : I/K/J/L + U (kick)
    BLUE[2] : (không hỗ trợ — sẽ thành bot)

Maps mặc định (HaxMods):
    1v1 → Winky's Futsal      HW=368, HH=171, goal_y=64
    2v2 → Felon Network v2    HW=520, HH=242, goal_y=76
    3v3 → Futsal 3v3 Classic  HW=630, HH=270, goal_y=80
    4v4 → VALN v4             HW=700, HH=320, goal_y=85
"""

from __future__ import annotations

import sys
import math
import argparse
import time
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError:
    print("Cần cài pygame:  pip install pygame")
    sys.exit(1)

from haxball_env import (
    HaxballEnv, Disc, DIR_MAP,
    BALL_R, PLYR_R, KICK_RANGE,
    PLYR_ACC, PLYR_KICK_ACC, KICK_STR,
    FRAME_SKIP,
)

# ─────────────────────────────────────────────────────────────────────────────
# HaxMods map presets
# ─────────────────────────────────────────────────────────────────────────────
MAPS: dict[str, dict] = {
    '1v1': dict(HW=368.0, HH=171.0, goal_y=64.0,  name="Winky's Futsal"),
    '2v2': dict(HW=520.0, HH=242.0, goal_y=76.0,  name="Felon Network v2"),
    '3v3': dict(HW=630.0, HH=270.0, goal_y=80.0,  name="Futsal 3v3 Classic"),
    '4v4': dict(HW=700.0, HH=320.0, goal_y=85.0,  name="VALN v4"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Key bindings  (tối đa 5 control sets, mỗi set = 1 người)
# ─────────────────────────────────────────────────────────────────────────────
# Mỗi set: (up, down, left, right, kick)
KEY_SETS = [
    # RED[0]
    (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_SPACE),
    # RED[1]
    (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RCTRL),
    # RED[2]
    (pygame.K_KP8, pygame.K_KP2, pygame.K_KP4, pygame.K_KP6, pygame.K_KP0),
    # BLUE[0]
    (pygame.K_t, pygame.K_g, pygame.K_f, pygame.K_h, pygame.K_r),
    # BLUE[1]
    (pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l, pygame.K_u),
]

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
C_BG          = (15,  20,  40)
C_FIELD       = (22,  92,  45)
C_LINE        = (255, 255, 255)
C_LINE_DIM    = (180, 180, 180)
C_GOAL_RED    = (220,  60,  60)
C_GOAL_BLUE   = ( 60, 100, 220)
C_BALL        = (240, 220, 100)
C_BALL_SHADOW = (180, 160,  60)
C_RED         = (230,  70,  70)
C_RED_OUT     = (255, 140, 140)
C_BLUE        = ( 60, 120, 240)
C_BLUE_OUT    = (120, 180, 255)
C_BOT_MARK    = (255, 255,  80)
C_HUD_BG      = ( 0,   0,   0, 180)
C_SCORE_RED   = (255, 100, 100)
C_SCORE_BLUE  = (100, 160, 255)
C_WHITE       = (255, 255, 255)
C_GRAY        = (160, 160, 160)
C_GOAL_FLASH  = (255, 240,  80)

WINDOW_W = 1200
WINDOW_H = 700
FPS      = 60
SCORE_TO_WIN = 5


# ─────────────────────────────────────────────────────────────────────────────
# Simple bot AI
# ─────────────────────────────────────────────────────────────────────────────
class SimpleBot:
    """
    Rule-based bot:
    - Luôn di chuyển về phía bóng (nếu bóng trong nửa sân mình hoặc mình gần nhất).
    - Nếu hold bóng → đá về phía goal đối thủ.
    - Thủ môn nếu bóng ở nửa sân đối thủ → đứng gần cột gôn.
    """

    def __init__(self, team_id: int, player_index: int, n_per_team: int):
        self.team_id      = team_id         # 1=RED, 2=BLUE
        self.player_index = player_index    # vị trí trong team (0=tiền đạo, n-1=thủ môn)
        self.n_per_team   = n_per_team
        self.attack_sign  = 1.0 if team_id == 1 else -1.0  # RED → +HW, BLUE → -HW

    def get_action(self, env: HaxballEnv) -> list[int]:
        """Trả về [dir_idx (0-8), kick (0/1)]."""
        ball       = env.ball
        my_players = env.red_players if self.team_id == 1 else env.blue_players
        ag         = my_players[self.player_index]
        HW, HH, gy = env.HW, env.HH, env.goal_y
        atk        = self.attack_sign

        # Tính khoảng cách bóng
        dx_b = ball.x - ag.x
        dy_b = ball.y - ag.y
        dist = math.hypot(dx_b, dy_b)

        # **Thủ môn** (player cuối team, nếu 2+) — ở lại vùng goal
        is_keeper = (self.n_per_team >= 2 and
                     self.player_index == self.n_per_team - 1)

        if is_keeper:
            # Di chuyển tới vị trí goal của mình
            goal_x    = -atk * HW * 0.75
            target_x  = goal_x
            target_y  = float(np.clip(ball.y, -gy * 0.8, gy * 0.8))
        else:
            # Tiền đạo/trung vệ: đuổi bóng
            target_x = ball.x
            target_y = ball.y

        # Vector hướng tới target
        tdx = target_x - ag.x
        tdy = target_y - ag.y
        td  = math.hypot(tdx, tdy)

        # Chọn hướng gần nhất trong 8+stay
        if td < 4.0:
            dir_idx = 0  # đứng yên
        else:
            nx, ny  = tdx / td, tdy / td
            best_i  = 0
            best_dot = -999.0
            for i in range(1, 9):  # bỏ qua 0=stay
                ddx, ddy = DIR_MAP[i]
                dlen     = math.hypot(ddx, ddy)
                dot      = (ddx / dlen) * nx + (ddy / dlen) * ny
                if dot > best_dot:
                    best_dot = dot
                    best_i   = i
            dir_idx = best_i

        # Kick: nếu trong tầm và hướng goal đối thủ thoáng
        surf_dist = max(0.0, dist - PLYR_R - BALL_R)
        kick = 0
        if surf_dist < KICK_RANGE + 2:
            kick = 1
            # Nếu là thủ môn thì chỉ kick để đẩy bóng ra xa
            if not is_keeper:
                # Hướng ball đến goal đối thủ
                goal_x = atk * HW
                gvx    = goal_x - ball.x
                gvy    = 0.0 - ball.y
                gd     = math.hypot(gvx, gvy)
                if gd > 0:
                    # Chọn hướng di chuyển về phía goal lúc kick
                    gnx, gny = gvx / gd, gvy / gd
                    best_i   = 0
                    best_dot = -999.0
                    for i in range(1, 9):
                        ddx, ddy = DIR_MAP[i]
                        dlen     = math.hypot(ddx, ddy)
                        dot      = (ddx / dlen) * gnx + (ddy / dlen) * gny
                        if dot > best_dot:
                            best_dot = dot
                            best_i   = i
                    dir_idx = best_i

        return [dir_idx, kick]


# ─────────────────────────────────────────────────────────────────────────────
# Renderer helpers
# ─────────────────────────────────────────────────────────────────────────────
class FieldView:
    """Chuyển tọa độ field → screen và vẽ mọi thứ."""

    def __init__(self, surface: pygame.Surface, env: HaxballEnv):
        self.surf = surface
        self.env  = env
        self._update_transform()

    def _update_transform(self):
        env        = self.env
        pad        = 60.0   # padding around field
        avail_w    = WINDOW_W
        avail_h    = WINDOW_H - 80   # 80px HUD at top
        scale_x    = (avail_w - 2 * pad) / (2 * env.HW)
        scale_y    = (avail_h - 2 * pad) / (2 * env.HH)
        self.scale = min(scale_x, scale_y)
        # Centre of field on screen
        self.cx    = WINDOW_W // 2
        self.cy    = 80 + avail_h // 2

    def f2s(self, x: float, y: float) -> tuple[int, int]:
        """Field coords → screen pixel."""
        s = self.scale
        return (int(self.cx + x * s), int(self.cy + y * s))

    def sr(self, r: float) -> int:
        return max(1, int(r * self.scale))

    def draw_field(self):
        env  = self.env
        s    = self.scale
        HW, HH, gy = env.HW, env.HH, env.goal_y

        # Field background
        tl = self.f2s(-HW, -HH)
        br = self.f2s( HW,  HH)
        pygame.draw.rect(self.surf, C_FIELD,
                         (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))

        # Centre circle
        r_c = int(min(HW, HH) * 0.2 * s)
        pygame.draw.circle(self.surf, C_LINE_DIM, (self.cx, self.cy), r_c, 2)
        pygame.draw.line(self.surf, C_LINE_DIM,
                         self.f2s(0, -HH), self.f2s(0, HH), 2)
        pygame.draw.circle(self.surf, C_LINE_DIM, (self.cx, self.cy), 4)

        # Field border
        pygame.draw.rect(self.surf, C_LINE,
                         (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]), 2)

        goal_depth = int(40 * s)

        # LEFT goal (Blue attacks here) — draw in RED colour
        gyl = self.f2s(-HW, -gy)
        gyl2 = self.f2s(-HW, gy)
        pygame.draw.line(self.surf, C_GOAL_RED, gyl, gyl2, 4)
        # goal back
        bl1 = (gyl[0] - goal_depth, gyl[1])
        bl2 = (gyl2[0] - goal_depth, gyl2[1])
        pygame.draw.line(self.surf, C_GOAL_RED, gyl,  bl1, 2)
        pygame.draw.line(self.surf, C_GOAL_RED, gyl2, bl2, 2)
        pygame.draw.line(self.surf, C_GOAL_RED, bl1,  bl2, 2)
        # pole circles
        pygame.draw.circle(self.surf, C_GOAL_RED, gyl,  6)
        pygame.draw.circle(self.surf, C_GOAL_RED, gyl2, 6)

        # RIGHT goal (Red attacks here) — draw in BLUE colour
        gyr  = self.f2s( HW, -gy)
        gyr2 = self.f2s( HW,  gy)
        pygame.draw.line(self.surf, C_GOAL_BLUE, gyr, gyr2, 4)
        br1  = (gyr[0] + goal_depth, gyr[1])
        br2  = (gyr2[0] + goal_depth, gyr2[1])
        pygame.draw.line(self.surf, C_GOAL_BLUE, gyr,  br1, 2)
        pygame.draw.line(self.surf, C_GOAL_BLUE, gyr2, br2, 2)
        pygame.draw.line(self.surf, C_GOAL_BLUE, br1,  br2, 2)
        pygame.draw.circle(self.surf, C_GOAL_BLUE, gyr,  6)
        pygame.draw.circle(self.surf, C_GOAL_BLUE, gyr2, 6)

    def draw_player(self, disc: Disc, color: tuple, outline: tuple,
                    label: str, is_human: bool):
        sx, sy = self.f2s(disc.x, disc.y)
        r      = self.sr(PLYR_R)
        pygame.draw.circle(self.surf, color,   (sx, sy), r)
        pygame.draw.circle(self.surf, outline, (sx, sy), r, 2)
        if not is_human:
            # Bot marker: small yellow dot
            pygame.draw.circle(self.surf, C_BOT_MARK, (sx, sy), max(3, r//4))
        # Label
        font = pygame.font.SysFont('Arial', max(9, r - 2), bold=True)
        txt  = font.render(label, True, C_WHITE)
        self.surf.blit(txt, (sx - txt.get_width()//2, sy - txt.get_height()//2))

    def draw_ball(self, disc: Disc):
        sx, sy = self.f2s(disc.x, disc.y)
        r      = self.sr(BALL_R)
        # Shadow
        pygame.draw.circle(self.surf, C_BALL_SHADOW, (sx + 2, sy + 2), r)
        pygame.draw.circle(self.surf, C_BALL,        (sx, sy), r)
        pygame.draw.circle(self.surf, (255, 255, 200), (sx, sy), r, 1)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(surf: pygame.Surface, score_red: int, score_blue: int,
             map_name: str, mode_str: str, step: int, max_steps: int,
             goal_flash: float, overtime: bool = False,
             time_limit_min: int = 3):
    # Background bar
    pygame.draw.rect(surf, (10, 10, 25), (0, 0, WINDOW_W, 80))
    pygame.draw.line(surf, (50, 50, 80), (0, 79), (WINDOW_W, 79), 1)

    big_font = pygame.font.SysFont('Arial', 42, bold=True)
    sm_font  = pygame.font.SysFont('Arial', 16)
    mid_font = pygame.font.SysFont('Arial', 22, bold=True)

    # Scores
    red_txt  = big_font.render(str(score_red),  True, C_SCORE_RED)
    blue_txt = big_font.render(str(score_blue), True, C_SCORE_BLUE)
    dash_txt = big_font.render(' — ', True, C_WHITE)
    total_w  = red_txt.get_width() + dash_txt.get_width() + blue_txt.get_width()
    sx       = (WINDOW_W - total_w) // 2
    surf.blit(red_txt,  (sx, 10))
    surf.blit(dash_txt, (sx + red_txt.get_width(), 10))
    surf.blit(blue_txt, (sx + red_txt.get_width() + dash_txt.get_width(), 10))

    # Map + Mode
    info_str = f"{map_name}  •  {mode_str}"
    info_txt = sm_font.render(info_str, True, C_GRAY)
    surf.blit(info_txt, (20, 10))

    # Timer bar
    bar_w = 300
    bar_x = (WINDOW_W - bar_w) // 2
    if overtime:
        # OVERTIME: nhấp nháy đỏ-cam
        blink = (int(pygame.time.get_ticks() / 400) % 2 == 0)
        bar_fill_c = (220, 80, 30) if blink else (255, 140, 60)
        pygame.draw.rect(surf, (40, 40, 60), (bar_x, 62, bar_w, 12), border_radius=4)
        pygame.draw.rect(surf, bar_fill_c,  (bar_x, 62, bar_w, 12), border_radius=4)
        ot_font = pygame.font.SysFont('Arial', 13, bold=True)
        ot_txt  = ot_font.render('⏱ OVERTIME', True, (255, 200, 80))
        surf.blit(ot_txt, (bar_x + bar_w // 2 - ot_txt.get_width() // 2, 62))
    else:
        ratio  = max(0.0, 1.0 - step / max(1, max_steps))
        ticks_left = max(0, max_steps - step)
        secs_left  = ticks_left // 60
        mins_left  = secs_left // 60
        secs_rem   = secs_left % 60
        pygame.draw.rect(surf, (40, 40, 60), (bar_x, 62, bar_w, 12), border_radius=4)
        fill_c = (80, 200, 80) if ratio > 0.4 else (220, 180, 60) if ratio > 0.2 else (220, 60, 60)
        pygame.draw.rect(surf, fill_c, (bar_x, 62, int(bar_w * ratio), 12), border_radius=4)
        # Hiển thị MM:SS bên cạnh bar
        t_font = pygame.font.SysFont('Arial', 13)
        t_txt  = t_font.render(f"{mins_left}:{secs_rem:02d} / {time_limit_min}:00", True, C_GRAY)
        surf.blit(t_txt, (bar_x + bar_w + 8, 62))

    # Controls hint
    hint = sm_font.render("ESC/Q: thoát  •  R: chơi lại  •  Bot = ●", True, (80, 80, 100))
    surf.blit(hint, (WINDOW_W - hint.get_width() - 20, 10))

    # Goal flash
    if goal_flash > 0:
        alpha = min(255, int(goal_flash * 350))
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((255, 240, 80, min(80, alpha)))
        surf.blit(overlay, (0, 0))
        gfont = pygame.font.SysFont('Arial', 72, bold=True)
        gt    = gfont.render('G O A L !', True, C_GOAL_FLASH)
        surf.blit(gt, ((WINDOW_W - gt.get_width()) // 2,
                       (WINDOW_H - gt.get_height()) // 2))


def draw_winner(surf: pygame.Surface, winner: str):
    """Màn hình kết thúc."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surf.blit(overlay, (0, 0))

    color = C_SCORE_RED if winner == 'RED' else C_SCORE_BLUE
    big   = pygame.font.SysFont('Arial', 80, bold=True)
    sm    = pygame.font.SysFont('Arial', 28)
    t1    = big.render(f"{winner} WINS!", True, color)
    t2    = sm.render("R: chơi lại  •  ESC/Q: thoát", True, C_WHITE)
    surf.blit(t1, ((WINDOW_W - t1.get_width()) // 2, WINDOW_H // 2 - 60))
    surf.blit(t2, ((WINDOW_W - t2.get_width()) // 2, WINDOW_H // 2 + 40))


# ─────────────────────────────────────────────────────────────────────────────
# Input handler → action
# ─────────────────────────────────────────────────────────────────────────────
def read_human_action(pressed: pygame.key.ScancodeWrapper, key_set: tuple) -> list[int]:
    up, down, left, right, kick = key_set
    dx = (1 if pressed[right] else 0) - (1 if pressed[left] else 0)
    dy = (1 if pressed[down]  else 0) - (1 if pressed[up]   else 0)

    # Map (dx, dy) → dir_idx
    dir_map_inv = {
        ( 0,  0): 0, ( 1,  0): 1, (-1,  0): 2,
        ( 0, -1): 3, ( 0,  1): 4,
        ( 1, -1): 5, (-1, -1): 6,
        ( 1,  1): 7, (-1,  1): 8,
    }
    dir_idx = dir_map_inv.get((dx, dy), 0)
    k       = 1 if pressed[kick] else 0
    return [dir_idx, k]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    print("\n" + "="*50)
    print("       HAXBALL — Chọn chế độ chơi")
    print("="*50)
    print("Chế độ: 1v1, 2v2, 3v3, 4v4")
    mode = input("Chọn chế độ [1v1]: ").strip().lower() or '1v1'
    if mode not in MAPS:
        print(f"Không nhận ra '{mode}', dùng 1v1.")
        mode = '1v1'

    n = int(mode[0])
    print(f"\nSố cầu thủ tối đa mỗi team: {n}")
    print(f"Controls RED  : KEY_SET 1=WASD+Space, 2=Arrows+RCtrl, 3=Numpad")
    print(f"Controls BLUE : KEY_SET 1=TFGH+R,     2=IJKL+U")

    red_h     = _get_int(f"Số người thật team RED  (0-{min(n,3)}) [1]: ", 0, min(n, 3), 1)
    blue_h    = _get_int(f"Số người thật team BLUE (0-{min(n,2)}) [0]: ", 0, min(n, 2), 0)
    win_score = _get_int("Số bàn thắng để kết thúc (1-20) [5]: ", 1, 20, SCORE_TO_WIN)
    time_lim  = _get_int("Giới hạn thời gian (phút, 1-6) [3]: ", 1, 6, 3)

    return mode, n, red_h, blue_h, win_score, time_lim


def _get_int(prompt: str, lo: int, hi: int, default: int) -> int:
    try:
        val = input(prompt).strip()
        return int(val) if val else default
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Main game loop
# ─────────────────────────────────────────────────────────────────────────────
def run(mode: str, n: int, red_humans: int, blue_humans: int,
        win_score: int, time_limit_min: int = 3):
    PLAY_SECONDS   = time_limit_min * 60
    PLAY_MAX_TICKS = PLAY_SECONDS * 60   # ticks @ 60 Hz

    mp      = MAPS[mode]
    env     = HaxballEnv(
        n_per_team = n,
        HW         = mp['HW'],
        HH         = mp['HH'],
        goal_y     = mp['goal_y'],
        spawn_mode = 'haxball',
        ep_seconds = PLAY_SECONDS,
        seed       = None,
    )
    # Override max_steps để timer bar hiển thị theo tick thật
    env.max_steps = PLAY_MAX_TICKS
    tick_count   = 0      # ticks trong hiệp hiện tại
    in_overtime  = False  # True sau khi hết giờ mà hoà

    # Build agent assignment:
    # agents order: red_0..red_{n-1}, blue_0..blue_{n-1}
    # Human slots: RED[0..red_humans-1], BLUE[0..blue_humans-1]
    # KEY_SET pool: RED humans take sets 0..2, BLUE humans take sets 3..4
    agent_is_human: list[bool]           = []
    agent_key_set:  list[Optional[tuple]] = []
    bots:           list[Optional[SimpleBot]] = []

    for i in range(n):  # RED
        is_h = i < red_humans and i < 3
        agent_is_human.append(is_h)
        agent_key_set.append(KEY_SETS[i] if is_h else None)
        bots.append(None if is_h else SimpleBot(1, i, n))

    for i in range(n):  # BLUE
        is_h = i < blue_humans and i < 2  # max 2 BLUE humans (key sets 3,4)
        agent_is_human.append(is_h)
        agent_key_set.append(KEY_SETS[3 + i] if is_h else None)
        bots.append(None if is_h else SimpleBot(2, i, n))

    mode_str = (f"RED: {red_humans} human + {n-red_humans} bot  |  "
                f"BLUE: {blue_humans} human + {n-blue_humans} bot")

    # Pygame init
    pygame.init()
    pygame.display.set_caption(f"HaxBall — {mode}  {mp['name']}")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock  = pygame.time.Clock()

    view        = FieldView(screen, env)
    score_red   = 0
    score_blue  = 0
    winner      = None
    goal_flash  = 0.0   # seconds remaining of flash
    FLASH_DUR   = 1.5

    obs, _ = env.reset()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0   # seconds

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Restart match
                    score_red  = 0
                    score_blue = 0
                    winner     = None
                    goal_flash = 0.0
                    env.reset()
                    tick_count  = 0
                    in_overtime = False

        if not running:
            break

        # ── Actions — 1 physics tick per frame = 60 Hz = tốc độ HaxBall thật ──
        if winner is None and goal_flash <= 0:
            pressed = pygame.key.get_pressed()
            actions = []
            for idx in range(env.n_agents):
                if agent_is_human[idx]:
                    actions.append(read_human_action(pressed, agent_key_set[idx]))
                else:
                    actions.append(bots[idx].get_action(env))

            # 1 tick/frame — đúng 60 ticks/s = HaxBall thật
            gr = env._tick(actions)
            if not in_overtime:
                tick_count += 1
            env.step_count = tick_count  # cho timer bar

            if gr == 1:    # RED scored (bóng vào gôn Blue)
                score_red   += 1
                goal_flash  = FLASH_DUR
                env.reset()
                # Timeline continues without reset
                if score_red >= win_score:
                    winner = 'RED'
            elif gr == -1: # BLUE scored (bóng vào gôn Red)
                score_blue  += 1
                goal_flash  = FLASH_DUR
                env.reset()
                # Timeline continues without reset
                if score_blue >= win_score:
                    winner = 'BLUE'
            elif not in_overtime and tick_count >= PLAY_MAX_TICKS:
                # Hết giờ chính thức
                if score_red == score_blue:
                    # Hoà → vào OVERTIME, không reset, chơi tiếp
                    in_overtime = True
                else:
                    # Có người đang dẫn → kết thúc
                    winner = 'RED' if score_red > score_blue else 'BLUE'

        # Goal flash countdown (still draw, no actions)
        if goal_flash > 0:
            goal_flash = max(0.0, goal_flash - dt)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        view.draw_field()

        # Players
        for idx, p in enumerate(env.red_players):
            lbl    = f"R{idx+1}"
            is_h   = agent_is_human[idx]
            view.draw_player(p, C_RED, C_RED_OUT, lbl, is_h)
        for idx, p in enumerate(env.blue_players):
            lbl    = f"B{idx+1}"
            is_h   = agent_is_human[n + idx]
            view.draw_player(p, C_BLUE, C_BLUE_OUT, lbl, is_h)

        # Ball
        view.draw_ball(env.ball)

        # HUD
        draw_hud(screen, score_red, score_blue, mp['name'], mode_str,
                 env.step_count, env.max_steps, goal_flash,
                 overtime=in_overtime, time_limit_min=time_limit_min)

        if winner is not None:
            draw_winner(screen, winner)

        pygame.display.flip()

    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='HaxBall Interactive Play')
    parser.add_argument('--mode',  type=str, default=None,
                        choices=['1v1', '2v2', '3v3', '4v4'],
                        help='Chế độ (1v1/2v2/3v3/4v4)')
    parser.add_argument('--red',   type=int, default=None,
                        help='Số người thật team RED')
    parser.add_argument('--blue',  type=int, default=None,
                        help='Số người thật team BLUE')
    parser.add_argument('--score', type=int, default=SCORE_TO_WIN,
                        help='Số bàn thắng để kết thúc (mặc định 5)')
    parser.add_argument('--time',  type=int, default=None,
                        choices=list(range(1, 7)),
                        help='Giới hạn thời gian mỗi hiệp (1-6 phút, mặc định 3)')
    parser.add_argument('--menu',  action='store_true',
                        help='Hiện menu hỏi thủ công')
    args = parser.parse_args()

    if args.menu:
        mode, n, red_humans, blue_humans, win_score, time_lim = interactive_menu()
    else:
        # Chạy thẳng với giá trị mặc định — không hỏi gì
        mode        = args.mode  if args.mode  is not None else '1v1'
        n           = int(mode[0])
        red_humans  = args.red   if args.red   is not None else 1
        blue_humans = args.blue  if args.blue  is not None else 0
        win_score   = args.score
        time_lim    = args.time  if args.time  is not None else 3

    print(f"\n▶  {mode}  •  {MAPS[mode]['name']}")
    print(f"   RED : {red_humans} người + {n-red_humans} bot")
    print(f"   BLUE: {blue_humans} người + {n-blue_humans} bot")
    print(f"   Thắng khi đạt {win_score} bàn  •  Giới hạn {time_lim} phút\n")
    run(mode, n, red_humans, blue_humans, win_score, time_lim)


if __name__ == '__main__':
    main()
