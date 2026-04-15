# D4-Adapted Image Transform Predict

Это отдельная адаптированная версия проекта под задачу предсказания преобразования между парой изображений в группе D4.

Что изменено:
- сохранена схема `pair encoder -> transformer decoder`;
- оставлен авторегрессивный вывод последовательности токенов;
- токены и target-последовательности переведены на канонические D4-слова над генераторами `e`, `r`, `s`;
- добавлен отдельный специальный токен `[NULL]` для пары изображений, не связанных допустимым преобразованием;
- augmentation scheduler полностью удален;
- доступны только `efficientnet_encoder` и `vit_encoder`.

## Структура

- основной train script: `run_train.py`
- основной конфиг по умолчанию: `configs/train_config_d4.yaml`
- альтернативный конфиг для ViT: `configs/train_config_d4_vit.yaml`

## Запуск обучения

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск с EfficientNet:

```bash
python run_train.py --data_path /path/to/domainnet_like_data
```

Запуск с ViT:

```bash
python run_train.py --data_path /path/to/domainnet_like_data --config configs/train_config_d4_vit.yaml
```

## Оценка двух моделей

Основной evaluation script: `run_eval_d4.py`.

Пример сравнения EfficientNet и ViT на отложенной выборке:

```bash
python run_eval_d4.py \
  --data_path /path/to/domainnet_like_data \
  --model efficientnet configs/train_config_d4.yaml outputs/checkpoints/d4_efficientnet/checkpoint_epoch_70.pth \
  --model vit configs/train_config_d4_vit.yaml outputs/checkpoints/d4_vit/checkpoint_epoch_70.pth \
  --batch_size 8 \
  --negative_pairs_per_image 1 \
  --output_dir outputs/eval_d4
```

Скрипт считает:
- teacher-forced token/sequence accuracy;
- greedy generation token/sequence accuracy;
- precision/recall/F1 по D4-классам и `[NULL]`;
- точность восстановленного applied-преобразования `g`;
- среднюю уверенность модели;
- метрики продолжения последовательности по длине prefix.

Результаты сохраняются в `outputs/eval_d4`:
- `metrics_overall.csv`;
- `metrics_by_domain.csv`;
- `metrics_by_class.csv`;
- `continuation_by_prefix.csv`;
- `continuation_by_domain_prefix.csv`;
- `plots/*.png`.

## Как кодируется задача

- положительная пара: `I2 = g(I1)`, где `g ∈ D4`
- target для положительной пары: каноническое короткое слово
  - `e`
  - `r`
  - `r r`
  - `r r r`
  - `s`
  - `s r`
  - `s r r`
  - `s r r r`
- отрицательная пара: target = `[NULL]`

Фактически модель видит последовательности вида:
- `[START] e [END]`
- `[START] r r [END]`
- `[START] s r r r [END]`
- `[START] [NULL] [END]`

## Ограничения и упрощения

- отрицательные пары формируются как случайные пары разных изображений из того же split; это минимально инвазивная адаптация без отдельной сложной проверки всех 8 преобразований;
- в конфиге по умолчанию `batch_size=8`, что ориентировано на ограничение по памяти; при наличии памяти можно поднять до `16`;
- pretrained-веса encoder'ов могут скачиваться библиотеками автоматически при первом запуске, если их нет в локальном кеше.
