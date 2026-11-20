# Финальный динозавр

DQN-агент для [T-Rex Runner](https://wayou.github.io/t-rex-runner/): бегает на
обычной игровой скорости, прыгает через кактусы и приседает под птицами.

## Установка

Python 3.10

```bash
pip install -r requirements.txt
python -m playwright install chromium
```



## Запуск

### Linux / macOS

```bash
chmod +x run_max.sh  
./run_max.sh
```

### Windows

Двойной клик по `ЗАПУСТИ_МАКСИМУМ.bat` либо:

```bat
python run_max.py
```

Принудительный старт: `--start-speed 13`.

Файл `dqn_dino_v2.pth` — обученная модель. `dino.py` содержит DQN и защитный
фильтр; обучение запускается командой `python dino.py`.
