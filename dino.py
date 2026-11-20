"""
DQN-агент для Chrome Dino Runner (https://wayou.github.io/t-rex-runner/).

Обучение:
    python dino.py
Просмотр жадной политики:
    python dino.py --eval

Агент получает 14 признаков (расстояние до препятствия, его размеры и
высоту, тип, скорость, время до столкновения, состояние динозавра) и
выбирает одно из трёх действий: бежать, прыгнуть, присесть.
Используются Double DQN и приоритезированная replay-память.

Поверх сети работает защитный фильтр (shielded RL): он считает безопасное
окно прыжка по реальной физике игры, планирует приземление перед следующим
препятствием и не даёт сети прыгать вхолостую. Отключить его можно флагом
--no-shield (чистый DQN). Для максимального забега используйте run_max.py.
"""

import argparse
import functools
import math
import os
import random
import shutil
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from playwright.sync_api import sync_playwright


# ---------- Настройки ----------

GAME_URL = "https://wayou.github.io/t-rex-runner/"
# Модели лежат рядом со скриптом — запуск работает из любого каталога,
# в том числе под Linux.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_ENVS = 1
EPISODES_PER_ENV = 10
MAX_STEPS_PER_EPISODE = 100000
STEP_DELAY = 0.02

# В обучении препятствия появляются раньше, чтобы не тратить 3 секунды
# каждого эпизода вхолостую. Динамика самих препятствий не меняется.
TRAIN_CLEAR_TIME = 1500
EVAL_CLEAR_TIME = 3000

GAMMA = 0.99
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
TARGET_UPDATE_EVERY = 750

# Prioritized Experience Replay
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 50_000

EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY = 25_000

MODEL_FILE = os.path.join(BASE_DIR, "dqn_dino_v2.pth")
BEST_MODEL_FILE = os.path.join(BASE_DIR, "dqn_dino_best.pth")
TRAIN_MODEL_FILE = os.path.join(BASE_DIR, "dqn_dino_train.pth")
SAVE_EVERY_STEPS = 5000

# Действия: 0 — бежать, 1 — прыгнуть, 2 — присесть
NUM_ACTIONS = 3

# Вектор состояния (14 признаков):
#  0  расстояние до препятствия
#  1  ширина препятствия (учитывает группы кактусов)
#  2  высота препятствия
#  3  вертикальная позиция препятствия (низ / середина / верх для птицы)
#  4  флаг «птица»
#  5  скорость игры
#  6  время до столкновения (секунды)
#  7  высота динозавра над землёй
#  8  флаг «в прыжке»
#  9  флаг «приседает»
# 10  вертикальная скорость прыжка
# 11  расстояние до следующего препятствия
# 12  остаток прыжка: насколько близко к пику/спуску
# 13  индикатор «препятствие отсутствует»
STATE_SIZE = 14

# Награды
STEP_REWARD = 0.05       # за выживший шаг
PASS_REWARD = 1.0        # за успешно пройденное препятствие
TIMEOUT_REWARD = 1.0     # за доживание до конца эпизода
DEATH_REWARD = -20.0     # за столкновение
JUMP_COST = 0.005        # крошечная плата за прыжок (меньше лишних прыжков)

# Флаги, чтобы вкладки не засыпали и их не останавливало по blur.
BROWSER_ARGS = [
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=CalculateNativeWinOcclusion",
]


def launch_browser(playwright, headless, args=None):
    """
    На Windows используем Edge, на Linux его обычно нет — тогда берём
    браузер Chromium, установленный через `playwright install chromium`.
    """
    kwargs = {"headless": headless, "args": BROWSER_ARGS if args is None else args}
    try:
        return playwright.chromium.launch(channel="msedge", **kwargs)
    except Exception:
        return playwright.chromium.launch(**kwargs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Геометрия прыжка (для защитного фильтра) ----------

DINO_WIDTH = 44.0
DINO_HEIGHT = 47.0
JUMP_FRAMES = 40  # запас: реальный прыжок длится ~34 кадра
LAND_FRAMES = 34


@functools.lru_cache(maxsize=512)
def jump_trajectory(speed, ground_y):
    """Высота динозавра по кадрам после нажатия Space (как в игре)."""
    y = float(ground_y)
    # В исходниках: INIITAL_JUMP_VELOCITY = -10, DROP_VELOCITY = -5.
    vel = -10.0 - float(speed) / 10.0
    path = []
    for _ in range(JUMP_FRAMES):
        y += round(vel)
        vel += 0.6
        if y < 30.0 and vel < -5.0:  # endJump() в исходниках игры
            vel = -5.0
        path.append(y)
        if y > ground_y:
            break
    return path


@functools.lru_cache(maxsize=2048)
def safe_jump_range(speed, width, top, ground_y):
    """
    Диапазон расстояний до препятствия (в px), при которых прыжок точно
    перелетает его. Учитываются реальные collision box'ы и длительность
    прыжка. Возвращает (мин, макс) или None, если прыжок невозможен.
    """
    path = jump_trajectory(speed, ground_y)
    good = []
    for gap in range(40, 700):
        first = max(1, math.ceil((gap - DINO_WIDTH) / speed))
        last = min(len(path), math.floor((gap + width) / speed))
        if first > last:
            continue
        if all(path[f - 1] + DINO_HEIGHT < top for f in range(first, last + 1)):
            good.append(gap)
    if not good:
        return None
    return good[0], good[-1]


# ---------- DQN ----------

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, shield=True):
        self.online = DQN().to(device)
        self.target = DQN().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)

        self.memory = deque(maxlen=REPLAY_SIZE)
        self.priorities = deque(maxlen=REPLAY_SIZE)
        self.steps = 0
        self.learns = 0
        self.shield = shield
        # Расстояние до препятствия в момент прыжка: позволяет понять,
        # что в воздухе мы всё ещё летим над «своим» препятствием.
        self.jump_start_gap = None
        # Последнее выбранное действие: нужно, чтобы отпустить ArrowDown
        # на один кадр перед приседанием после speed-drop в воздухе.
        self.last_action = 0

    # --- epsilon-greedy ---

    def epsilon(self):
        fraction = min(self.steps / EPSILON_DECAY, 1.0)
        return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)

    def _allowed_actions(self, state):
        # В воздухе повторный прыжок игнорируется, а приседание вызывает
        # резкое падение и ломает соответствие между действием и состоянием.
        # Оставляем только «продолжать прыжок», чтобы ускорить обучение.
        if state[8] > 0.5:
            return [0]
        return list(range(NUM_ACTIONS))

    @staticmethod
    def _need(obstacle):
        """Что нужно сделать с препятствием: jump / duck / none."""
        if obstacle is None:
            return "none"
        if obstacle["bird"]:
            if obstacle["y"] < 65.0:
                return "none"      # высокая птица летит над головой
            if obstacle["y"] < 90.0:
                return "duck"      # средняя птица — только приседание
            return "jump"          # низкая птица
        return "jump"

    def _shield_action(self, state, action, raw=None):
        """
        Защитный фильтр (shielded RL) поверх DQN.

        Сеть по-прежнему предлагает действие, но фильтр проверяет его по
        реальной геометрии игры и планирует посадку:

          - прыжок начинается в рассчитанном по физике безопасном окне;
          - если второе препятствие близко, прыжок откладывается к концу
            окна, чтобы динозавр приземлился до следующего препятствия;
          - после просчёта текущего препятствия в воздухе включается
            speed-drop (ArrowDown) — быстрое приземление;
          - средняя птица — приседание, высокая — никаких действий.
        """
        if not self.shield:
            return action

        if raw is None and state[13] < 0.5:
            return action

        if raw is None:
            # Резервный вариант (используется только если Playwright не дал
            # подробную информацию об игре).
            gap = state[0] * 600.0
            speed = state[5] * 13.0
            width = state[1] * 100.0
            obstacle_y = state[3] * 120.0
            if state[4] > 0.5 and 65.0 <= obstacle_y < 90.0:
                return 2 if gap < 40.0 + speed * 4.0 else 0
            if state[4] > 0.5 and obstacle_y < 65.0:
                return 0
            if state[8] > 0.5:
                return 0
            jump_at = 40.0 + speed * 6.0 + width * 0.25
            return 1 if gap < jump_at else 0

        speed = max(float(raw.get("speed") or 1.0), 1.0)
        dino_x = float(raw.get("dinoX") or 0.0)
        ground = float(raw.get("groundY") or 93.0)
        jumping = bool(raw.get("jumping"))
        ducking = bool(raw.get("ducking"))

        def obstacle(prefix):
            x = raw.get(prefix + "X")
            if x is None:
                return None
            y = float(raw.get(prefix + "Y") or 105.0)
            bird = bool(raw.get(prefix + "Bird"))
            # Collision box'ы птицы начинаются на 8 px ниже yPos.
            return {
                "x": float(x),
                "gap": float(x) - dino_x,
                "w": float(raw.get(prefix + "W") or 0.0),
                "y": y,
                "bird": bird,
                "top": y + (8.0 if bird else 0.0),
            }

        cur = obstacle("cur")
        nxt = obstacle("next")

        # Нет препятствий — бежим. Сеть иногда прыгает вхолостую, из-за чего
        # оказывается в воздухе в момент появления следующего препятствия.
        if cur is None and nxt is None:
            return 0

        # Уже в прыжке. Траекторию нельзя обрывать, только пока мы летим
        # над тем препятствием, ради которого прыгали. Как только оно позади
        # (или в списке появилось новое), жмём ArrowDown и садимся быстрее.
        if jumping:
            start_gap = self.jump_start_gap
            if (
                start_gap is not None
                and cur is not None
                and cur["gap"] <= start_gap + 5.0
                and cur["gap"] + cur["w"] >= -5.0
            ):
                return 0
            return 2

        self.jump_start_gap = None

        if ducking:
            if cur is not None and self._need(cur) == "duck" and \
                    cur["gap"] + cur["w"] >= -5.0:
                return 2
            return 0

        if cur is None:
            return 0

        need = self._need(cur)

        if need == "none":
            return 0

        if need == "duck":
            duck_at = 45.0 + speed * 4.0
            if cur["gap"] >= duck_at:
                return 0
            # Если ArrowDown был зажат ещё в воздухе (speed-drop), то после
            # приземления игра не вызовет startDuck() — отпускаем клавишу
            # на один кадр, иначе динозавр останется стоять и столкнётся.
            if not ducking and self.last_action == 2:
                return 0
            return 2

        window = safe_jump_range(speed, cur["w"], cur["top"], ground)
        if window is None:
            # Такой ситуации в игре не бывает, но лучше присесть, чем
            # столкнуться наверняка.
            return 2
        low, high = window
        # держимся ближе к середине безопасного окна: модель физики слегка
        # расходится с браузером, а ранний прыжок опаснее позднего.
        target = low + 0.45 * (high - low)

        if nxt is not None and nxt["gap"] > cur["gap"] + 5.0:
            spacing = nxt["gap"] - cur["gap"]
            next_need = self._need(nxt)
            if next_need == "jump":
                next_window = safe_jump_range(
                    speed, nxt["w"], nxt["top"], ground
                )
                land_gap = (next_window[0] + 25.0) if next_window else 150.0
            else:
                # К следующему препятствию надо уже бежать/приседать.
                land_gap = 130.0
            # После приземления (≈36 кадров) следующее препятствие должно
            # оставаться впереди с запасом на новое действие.
            limit = LAND_FRAMES * speed + land_gap - spacing - 20.0
            target = min(target, limit)

        target = max(target, low + 5.0)
        if cur["gap"] <= target:
            self.jump_start_gap = cur["gap"]
            return 1
        return 0

    def choose_action(self, state, training=True, raw=None):
        self.steps += 1
        allowed = self._allowed_actions(state)

        if training and random.random() < self.epsilon():
            action = random.choice(allowed)
        else:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = self.online(state_tensor)[0]

            mask = torch.full((NUM_ACTIONS,), float("-inf"), device=device)
            for allowed_action in allowed:
                mask[allowed_action] = 0.0

            action = int((q_values + mask).argmax().item())

        result = self._shield_action(state, action, raw)
        self.last_action = result
        return result

    # --- replay memory ---

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def _beta(self):
        fraction = min(self.steps / PER_BETA_FRAMES, 1.0)
        return PER_BETA_START + fraction * (1.0 - PER_BETA_START)

    # --- Double DQN + PER ---

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        priorities = np.asarray(self.priorities, dtype=np.float32)
        probs = np.power(priorities, PER_ALPHA)
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probs)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        current_q = self.online(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        # Веса важности для смещения, которое вносит приоритетный сэмплинг.
        weights = np.power(len(self.memory) * probs[indices], -self._beta())
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        loss_per_item = nn.functional.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )
        loss = (loss_per_item * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        td_error = (target_q - current_q).abs().detach().cpu().numpy() + 1e-6
        for index, error in zip(indices, td_error):
            self.priorities[index] = float(error)

        self.learns += 1
        if self.steps % TARGET_UPDATE_EVERY == 0:
            self.target.load_state_dict(self.online.state_dict())

    # --- модель ---

    def save(self, path=None):
        path = path or MODEL_FILE
        torch.save(
            {
                "model": self.online.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
            },
            path,
        )
        print(f"Модель сохранена: {path}")

    def load(self, path=None):
        path = path or MODEL_FILE
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
            # Старый формат — просто state_dict.
            if isinstance(payload, dict) and "model" in payload:
                self.online.load_state_dict(payload["model"])
                if "optimizer" in payload:
                    self.optimizer.load_state_dict(payload["optimizer"])
                self.steps = int(payload.get("steps", 0))
            else:
                self.online.load_state_dict(payload)
            self.target.load_state_dict(self.online.state_dict())
            print(f"Загружена модель: {path} (шагов: {self.steps})")
            return True
        except FileNotFoundError:
            print("Модель не найдена — обучение начнётся с нуля.")
        except Exception as error:
            print(f"Не удалось загрузить модель ({error}). Обучение начнётся с нуля.")
        return False


# ---------- Работа с игрой ----------

STATE_JS = """() => {
    const r = window.Runner && window.Runner.instance_;
    if (!r || !r.tRex || !r.horizon) return {status: 'NO_RUNNER'};
    if (r.crashed) return {status: 'DEAD', score: Math.ceil(r.distanceRan || 0)};
    if (!r.playing) return {status: 'NOT_PLAYING'};

    const d = r.tRex;
    const list = r.horizon.obstacles || [];
    const o = list[0];
    const n = list[1];

    const out = {
        status: 'OK',
        speed: Number(r.currentSpeed) || 0.1,
        score: Math.ceil(r.distanceRan || 0),
        dinoX: Number(d.xPos) || 0,
        dinoY: Number(d.yPos) || 0,
        groundY: Number(d.groundYPos) || 93,
        jumping: d.jumping ? 1 : 0,
        ducking: d.ducking ? 1 : 0,
        vel: Number(d.jumpVelocity) || 0,
        obstacleX: o ? Number(o.xPos) : null,
        obstacleW: o ? Number(o.width) : 0,
        obstacleY: o ? Number(o.yPos) : 0,
        isBird: o ? (String((o.typeConfig || {}).type || '').toUpperCase().indexOf('PTERODACTYL') >= 0 ? 1 : 0) : 0,
        curX: o ? Number(o.xPos) : null,
        curW: o ? Number(o.width) : 0,
        curY: o ? Number(o.yPos) : 0,
        curBird: o ? (String((o.typeConfig || {}).type || '').toUpperCase().indexOf('PTERODACTYL') >= 0 ? 1 : 0) : 0,
        nextX: n ? Number(n.xPos) : null,
        nextW: n ? Number(n.width) : 0,
        nextY: n ? Number(n.yPos) : 0,
        nextBird: n ? (String((n.typeConfig || {}).type || '').toUpperCase().indexOf('PTERODACTYL') >= 0 ? 1 : 0) : 0
    };

    if (o) {
        const cfg = o.typeConfig || {};
        out.width = Number(o.width) || Number(cfg.width) || 0;
        out.height = Number(cfg.height) || 0;
        out.obstacleY = Number(o.yPos) || 0;
        out.isBird = String(cfg.type || '').toUpperCase().indexOf('PTERODACTYL') >= 0 ? 1 : 0;
    } else {
        out.width = 0;
        out.height = 0;
        out.obstacleY = 0;
        out.isBird = 0;
    }
    return out;
}"""


def clip(value, low=0.0, high=2.0):
    return float(np.clip(value, low, high))


def get_state(page):
    """Возвращает (вектор состояния, сырая информация об игре)."""
    try:
        data = page.evaluate(STATE_JS)
    except Exception as error:
        return None, {"status": f"ERROR: {error}"}

    if data.get("status") != "OK":
        return None, data

    speed = max(float(data["speed"]), 0.1)

    obstacle_x = data.get("obstacleX")
    gap = 1800.0 if obstacle_x is None else max(float(obstacle_x) - float(data["dinoX"]), 0.0)
    next_x = data.get("nextX")
    next_gap = 1800.0 if next_x is None else max(float(next_x) - float(data["dinoX"]), 0.0)

    # px/кадр -> px/сек при 60 FPS.
    time_to_obstacle = gap / (speed * 60.0)
    dino_height = float(data["groundY"]) - float(data["dinoY"])
    has_obstacle = 0.0 if obstacle_x is None else 1.0
    vel = float(data["vel"])
    # Чем ближе velocity к нулю после отрицательного пика, тем ближе спуск.
    jump_phase = clip((vel + 10.0) / 10.0, -1.0, 2.0)

    state = np.array([
        clip(gap / 600.0),
        clip(float(data["width"]) / 100.0),
        clip(float(data["height"]) / 60.0),
        clip(float(data["obstacleY"]) / 120.0),
        float(data["isBird"]),
        clip(speed / 13.0),
        clip(time_to_obstacle / 1.5),
        clip(dino_height / 100.0),
        float(data["jumping"]),
        float(data["ducking"]),
        clip(vel / 12.0, -1.5, 1.5),
        clip(next_gap / 600.0),
        jump_phase,
        has_obstacle,
    ], dtype=np.float32)

    return state, data


def release_controls(page):
    try:
        page.keyboard.up("Space")
        page.keyboard.up("ArrowDown")
    except Exception:
        pass


def apply_action(page, old_action, new_action):
    """ArrowDown держим нажатым, пока выбрано приседание."""
    if old_action == new_action:
        return

    if old_action == 2:
        page.keyboard.up("ArrowDown")

    if new_action == 1:
        page.keyboard.press("Space")
    elif new_action == 2:
        page.keyboard.down("ArrowDown")


def prepare_page(page, clear_time, initial_speed=None):
    page.goto(GAME_URL, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => !!(window.Runner && Runner.instance_ && Runner.instance_.horizon)",
        timeout=20000,
    )
    # Игра по умолчанию ставит на паузу все вкладки, кроме активной.
    # Отключаем этот обработчик ДО первого запуска игры.
    page.evaluate("""([clearTime, initialSpeed]) => {
        const r = Runner.instance_;
        r.onVisibilityChange = function () {};
        r.loadSounds = function () {};
        r.playSound = function () {};
        r.config.CLEAR_TIME = clearTime;
        if (initialSpeed) {
            r.config.SPEED = initialSpeed;
            r.setSpeed(initialSpeed);
        }
    }""", [clear_time, initial_speed])


def start_or_restart_game(page):
    release_controls(page)

    needs_start = False
    try:
        needs_start = bool(page.evaluate("""() => {
            const r = window.Runner && Runner.instance_;
            if (!r) return false;
            r.onVisibilityChange = function () {};
            r.playSound = function () {};
            // stop() нужен, чтобы restart() сработал и во время активной игры
            // (например, когда эпизод закончился по лимиту шагов).
            r.stop();
            r.restart();
            r.paused = false;
            // Самый первый запуск требует прыжка: он включает intro,
            // после которого начинает двигаться горизонт.
            return !r.activated;
        }"""))
    except Exception:
        pass

    time.sleep(0.05)

    if needs_start:
        page.keyboard.press("Space")
        time.sleep(0.1)

    for _ in range(6):
        state, info = get_state(page)
        if state is not None:
            return state, info
        try:
            page.keyboard.press("Space")
        except Exception:
            pass
        time.sleep(0.2)

    return None, info


def check_passed(previous_x, previous_width, current_x):
    """Определяем, что препятствие осталось позади."""
    if previous_x is None:
        return False
    if current_x is None:
        return True
    if current_x > previous_x + 20.0:
        return True
    if current_x + previous_width < 0.0:
        return True
    return False


# ---------- Защита лучшей модели ----------


def ensure_best_model():
    """Фиксируем текущие хорошие веса, чтобы обучение не могло их испортить."""
    if os.path.exists(MODEL_FILE) and not os.path.exists(BEST_MODEL_FILE):
        shutil.copy2(MODEL_FILE, BEST_MODEL_FILE)
        print(f"Лучшая модель зафиксирована: {BEST_MODEL_FILE}")


def measure_model(path, seconds=25):
    """Быстрая жадная проверка модели: (лучший счёт, число столкновений)."""
    agent = Agent()
    agent.load(path)
    best = 0
    collisions = 0
    with sync_playwright() as p:
        browser = launch_browser(p, headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        prepare_page(page, EVAL_CLEAR_TIME)
        state, state_info = start_or_restart_game(page)
        action = 0
        current = 0
        start = time.time()
        try:
            while time.time() - start < seconds:
                if state is None:
                    collisions += 1
                    state, state_info = start_or_restart_game(page)
                    action = 0
                    current = 0
                    continue
                new_action = agent.choose_action(
                    state, training=False, raw=state_info
                )
                apply_action(page, action, new_action)
                action = new_action
                time.sleep(STEP_DELAY)
                state, info = get_state(page)
                state_info = info if state is not None else {}
                current = max(current, int(info.get("score") or 0))
                best = max(best, current)
        except Exception as error:
            print(f"Ошибка при проверке модели {path}: {error}")
        finally:
            browser.close()
    return best, collisions


def promote_if_better():
    """Повышаем новую модель только если она реально лучше зафиксированной."""
    if not os.path.exists(TRAIN_MODEL_FILE):
        return
    if not os.path.exists(BEST_MODEL_FILE):
        shutil.copy2(TRAIN_MODEL_FILE, BEST_MODEL_FILE)
        shutil.copy2(TRAIN_MODEL_FILE, MODEL_FILE)
        return

    old_score, old_deaths = measure_model(BEST_MODEL_FILE)
    new_score, new_deaths = measure_model(TRAIN_MODEL_FILE)
    print(
        f"Проверка качества: была={old_score} очков/{old_deaths} смертей; "
        f"стала={new_score} очков/{new_deaths} смертей"
    )
    if new_score > old_score and new_deaths <= old_deaths:
        shutil.copy2(TRAIN_MODEL_FILE, BEST_MODEL_FILE)
        shutil.copy2(TRAIN_MODEL_FILE, MODEL_FILE)
        print("Новая модель лучше — установлена как основная.")
    else:
        print("Новая модель не лучше — оставлена проверенная.")


# ---------- Обучение ----------

def train(args):
    ensure_best_model()
    agent = Agent(shield=not args.no_shield)
    start_path = BEST_MODEL_FILE if os.path.exists(BEST_MODEL_FILE) else MODEL_FILE
    agent.load(start_path)
    if args.steps is not None:
        # Позволяет продолжить обучение старой модели, задав текущий прогресс.
        agent.steps = args.steps

    clear_time = args.clear_time if args.clear_time is not None else TRAIN_CLEAR_TIME

    with sync_playwright() as p:
        browser = launch_browser(p, headless=args.headless)

        pages = []
        for i in range(args.envs):
            page = browser.new_page(viewport={"width": 1000, "height": 700})
            prepare_page(page, clear_time, args.train_speed)
            pages.append(page)
            print(f"Открыта игра {i + 1}/{args.envs}.")

        time.sleep(0.5)

        states = [None] * args.envs
        infos = [{} for _ in range(args.envs)]
        actions = [0] * args.envs
        steps_in_episode = [0] * args.envs
        episodes_done = [0] * args.envs
        last_scores = [0] * args.envs
        previous_x = [None] * args.envs
        previous_width = [0.0] * args.envs
        best_score = 0

        for i, page in enumerate(pages):
            states[i], infos[i] = start_or_restart_game(page)
            previous_x[i] = None

        total_steps = 0

        while any(ep < args.episodes for ep in episodes_done):
            active = [i for i in range(args.envs) if episodes_done[i] < args.episodes]

            # 1. Выбираем и выполняем действия.
            for i in active:
                if states[i] is None:
                    states[i], infos[i] = start_or_restart_game(pages[i])
                    actions[i] = 0
                    previous_x[i] = None
                    if states[i] is None:
                        continue

                new_action = agent.choose_action(
                    states[i], training=True, raw=infos[i]
                )
                apply_action(pages[i], actions[i], new_action)
                actions[i] = new_action

            time.sleep(STEP_DELAY)

            # 2. Собираем переходы.
            for i in active:
                if states[i] is None:
                    continue

                next_state, info = get_state(pages[i])
                crashed = info.get("status") == "DEAD"
                done = next_state is None
                timeout = steps_in_episode[i] + 1 >= args.max_steps

                passed = check_passed(
                    previous_x[i], previous_width[i], info.get("obstacleX")
                )
                previous_x[i] = info.get("obstacleX")
                previous_width[i] = float(info.get("obstacleW") or 0.0)

                if crashed:
                    reward = DEATH_REWARD
                elif timeout:
                    reward = TIMEOUT_REWARD
                elif done:
                    reward = DEATH_REWARD
                else:
                    reward = STEP_REWARD
                    if passed:
                        reward += PASS_REWARD
                    if actions[i] == 1:
                        reward -= JUMP_COST

                steps_in_episode[i] += 1
                total_steps += 1
                last_scores[i] = max(last_scores[i], int(info.get("score") or 0))

                next_for_memory = (
                    np.zeros(STATE_SIZE, dtype=np.float32) if done else next_state
                )
                agent.remember(states[i], actions[i], reward, next_for_memory, done)
                agent.learn()

                if done or timeout:
                    score = last_scores[i]
                    best_score = max(best_score, score)
                    print(
                        f"Игра {i + 1}: эпизод {episodes_done[i] + 1}/{args.episodes}, "
                        f"шагов: {steps_in_episode[i]}, очков: {score}, "
                        f"лучший: {best_score}, epsilon: {agent.epsilon():.3f}"
                    )

                    episodes_done[i] += 1
                    steps_in_episode[i] = 0
                    last_scores[i] = 0
                    actions[i] = 0
                    previous_x[i] = None
                    previous_width[i] = 0.0
                    release_controls(pages[i])

                    if episodes_done[i] < args.episodes:
                        states[i], infos[i] = start_or_restart_game(pages[i])
                        if states[i] is None:
                            print(f"Перезапуск игры {i + 1} не удался, обновляю вкладку.")
                            pages[i].reload(wait_until="domcontentloaded")
                            time.sleep(1)
                            prepare_page(pages[i], clear_time, args.train_speed)
                            states[i], infos[i] = start_or_restart_game(pages[i])
                    else:
                        states[i] = None
                else:
                    states[i] = next_state
                    infos[i] = info

            if total_steps % 1000 == 0:
                print(
                    f"Всего переходов: {total_steps}; epsilon: {agent.epsilon():.3f}; "
                    f"обучений: {agent.learns}; память: {len(agent.memory)}"
                )

            if total_steps > 0 and total_steps % SAVE_EVERY_STEPS == 0:
                agent.save(TRAIN_MODEL_FILE)

        agent.save(TRAIN_MODEL_FILE)
        print("Обучение завершено.")

    # Проверка качества запускает отдельный браузер, поэтому она должна
    # выполняться после закрытия Playwright из обучения.
    promote_if_better()

    if not args.no_prompt:
        input("Нажми Enter, чтобы закрыть Edge...")


# ---------- Режим просмотра (жадная политика) ----------

def evaluate(args):
    agent = Agent(shield=not args.no_shield)
    if not agent.load():
        print("Внимание: модель не загружена, политика случайна.")

    clear_time = args.clear_time if args.clear_time is not None else EVAL_CLEAR_TIME

    with sync_playwright() as p:
        browser = launch_browser(p, headless=args.headless)
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        prepare_page(page, clear_time, args.train_speed)
        state, state_info = start_or_restart_game(page)
        action = 0
        last_score = 0
        best = 0
        deaths = 0
        last_info = {}
        history = []
        action_counts = [0, 0, 0]
        start_time = time.time()

        try:
            while args.seconds is None or time.time() - start_time < args.seconds:
                if state is None:
                    release_controls(page)
                    state, state_info = start_or_restart_game(page)
                    action = 0
                    if state is not None:
                        deaths += 1
                        print(f"Столкновение: очков {last_score}, лучший {best}")
                        if args.debug:
                            print(
                                "  перед смертью: "
                                f"тип={last_info.get('isBird')} "
                                f"gap={(float(last_info.get('obstacleX') or 0) - float(last_info.get('dinoX') or 0)):.1f} "
                                f"w={last_info.get('obstacleW')} "
                                f"y={last_info.get('obstacleY')} "
                                f"speed={last_info.get('speed'):.2f} "
                                f"jump={last_info.get('jumping')} "
                                f"duck={last_info.get('ducking')}"
                            )
                            print("  последние шаги (gap/w/y/тип/прыжок/действие):")
                            for item in history[-60:]:
                                print("    " + item)
                    history = []
                    last_score = 0
                    continue

                new_action = agent.choose_action(
                    state, training=False, raw=state_info
                )
                apply_action(page, action, new_action)
                action_counts[new_action] += 1
                action = new_action
                if args.debug:
                    history.append(
                        f"gap={state[0]*600:6.1f} w={state[1]*100:5.1f} "
                        f"y={state[3]*120:5.1f} bird={int(state[4])} "
                        f"rg={(float(state_info.get('curX') or 0)-float(state_info.get('dinoX') or 0)):6.1f} "
                        f"next={state[11]*600:6.1f} h={state[7]*100:5.1f} "
                        f"jump={int(state[8])} duck={int(state[9])} "
                        f"act={new_action} vel={state[10]*12:6.1f}"
                    )
                    if len(history) > 200:
                        history = history[-100:]
                time.sleep(STEP_DELAY)

                next_state, info = get_state(page)
                score = int(info.get("score") or 0)
                if info.get("status") == "OK":
                    last_info = info
                last_score = max(last_score, score)
                best = max(best, last_score)
                if next_state is None:
                    release_controls(page)
                state = next_state
                state_info = info if state is not None else {}
        except KeyboardInterrupt:
            print("\nОстановлено.")
        finally:
            print(
                f"Итог: последний счёт {last_score}, лучший {best}, "
                f"столкновений {deaths}"
            )
            print(
                "Действия: бег={0}, прыжок={1}, приседание={2}".format(*action_counts)
            )
            browser.close()


def main():
    parser = argparse.ArgumentParser(description="DQN для Chrome Dino Runner")
    parser.add_argument("--eval", action="store_true", help="только жадная игра")
    parser.add_argument("--envs", type=int, default=NUM_ENVS)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ENV)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--clear-time", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--seconds", type=float, default=None,
                        help="в режиме --eval: сколько секунд играть")
    parser.add_argument("--steps", type=int, default=None,
                        help="переопределить счётчик шагов (для дообучения)")
    parser.add_argument("--debug", action="store_true",
                        help="печатать детали столкновений в режиме --eval")
    parser.add_argument("--no-shield", action="store_true",
                        help="отключить защитный фильтр (чистый DQN)")
    parser.add_argument("--train-speed", type=float, default=None,
                        help="стартовая скорость при обучении (куррикулум)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
