# Финальный динозавр

DQN-агент для [T-Rex Runner](https://wayou.github.io/t-rex-runner/): бегает на
обычной игровой скорости, прыгает через кактусы и приседает под птицами.
В репозитории лежит только то, что нужно для запуска.

## Установка

Python 3.10+ (рекомендуется 3.11/3.12).

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

На Windows вместо `python` используйте `py -3`, браузер Edge находится автоматически,
но Chromium из Playwright тоже подойдёт.

## Запуск

### Linux / macOS

```bash
chmod +x run_max.sh   # один раз после распаковки
./run_max.sh
```

### Windows

Двойной клик по `ЗАПУСТИ_МАКСИМУМ.bat` либо:

```bat
python run_max.py
```

## Полезные флаги

```bash
./run_max.sh --seconds 600    # закончить через 10 минут
./run_max.sh --headless       # без окна браузера, максимум FPS
./run_max.sh --debug          # телеметрия и детали столкновений
```

Скорость не форсируется: старт 6 (как в игре), дальше естественный разгон.
Принудительный старт: `--start-speed 13`.

Файл `dqn_dino_v2.pth` — обученная модель. `dino.py` содержит DQN и защитный
фильтр; обучение запускается командой `python dino.py`.
