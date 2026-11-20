"""
Максимально жёсткий забег DQN-динозавра.

Запуск:
    python run_max.py                 # видимый браузер, бесконечный забег
    python run_max.py --seconds 600   # 10 минут и автоматический выход
    python run_max.py --headless      # без окна браузера (максимум FPS)

Используется проверенная модель dqn_dino_v2.pth и защитный фильтр.
Скорость НЕ форсируется: старт как в игре по умолчанию (6), дальнейший
разгон — естественный, как в оригинальном T-Rex Runner.
"""

import argparse

from dino import evaluate


def main():
    parser = argparse.ArgumentParser(description="Максимальный забег Dino DQN")
    parser.add_argument(
        "--start-speed",
        type=float,
        default=None,
        help="принудительная стартовая скорость (по умолчанию — как в игре)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="сколько секунд играть (по умолчанию — до Ctrl+C)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="не показывать окно браузера",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="печатать телеметрию и детали столкновений",
    )
    parser.add_argument(
        "--no-shield",
        action="store_true",
        help="отключить защитный фильтр (не рекомендуется)",
    )
    opts = parser.parse_args()

    args = argparse.Namespace(
        eval=True,
        envs=1,
        episodes=1,
        max_steps=100000,
        clear_time=None,
        headless=opts.headless,
        no_prompt=True,
        seconds=opts.seconds,
        steps=None,
        debug=opts.debug,
        no_shield=opts.no_shield,
        train_speed=opts.start_speed,
    )

    if opts.start_speed is None:
        print("Запуск: обычная скорость игры (6), щит включён.")
    else:
        print(f"Запуск: стартовая скорость {opts.start_speed}, щит включён.")
    try:
        evaluate(args)
    except KeyboardInterrupt:
        print("\nЗабег остановлен.")


if __name__ == "__main__":
    main()
