# agentic-course-itmo-llm-prediction: AR-09 на арене

Репозиторий трека B: навык, выращенный на арене. Тема — **AR-09, прогноз длины ответа LLM**. Нужно по паре «запрос × модель» предсказать число выходных токенов и интервал 10–90 %, а затем упаковать найденное решение в агентский навык.

ТЗ: [B-arena-skill.md](https://github.com/deeppavlov/agentic-course-itmo/blob/main/projects/B-arena-skill.md). Реестр тем: [arena-registry.md](https://github.com/deeppavlov/agentic-course-itmo/blob/main/projects/arena-registry.md). Рабочая копия кейса для арены находится в `arena-main/cases/ar09`; этот репозиторий хранит описание, данные и проверяемые артефакты кейса.

## Статус

Подготовка к ЧП1 завершена, кампания ещё не запускалась: в журнале **0 из 40** решений. Соответствие требованиям ЧП1 и оставшиеся внешние действия — в [`docs/checkpoint-1.md`](docs/checkpoint-1.md).

## Итог в одной таблице

| | Train | Public | Private |
| --- | --- | --- | --- |
| Baseline: глобальные квантили | 1.5089 | 1.4792 | 1.4848 |
| Выращенное решение | — | — | — |
| Классический подход: квантили по модели | — | MAE 0.734 | — |
| Классический подход: модель × источник | — | MAE 0.635 | — |

Значения baseline — полная метрика `ar09-quantile-v1` в шкале `log1p`, посчитанная по батчам. Для сравнения MAE на public: baseline `0.743`, оракул по собственным ответам строки `0.152`. Лимит предложенных решений: **40**. Кандидатов в журнале: **0**. AUC кривой: **—**, кампания не запущена.

## Задача и метрика

Вход батча: `{"rows": [{"id", "prompt", "model", "input_tokens"}, ...]}`. Выход: `{"<id>": {"pred", "lo", "hi"}, ...}` — точечный прогноз и интервал 10–90 % в токенах. `pred` — медиана, `lo`/`hi` — квантили 0.1/0.9.

Метрика минимизируется: `MAE(pred) + 2·pinball_0.1(lo) + 2·pinball_0.9(hi)` в шкале `log1p(токенов)`, среднее по строкам. Невалидная строка получает штраф `40`. Метрика без LLM, решатель тоже без LLM.

## Структура

```text
docs/domain.md          обзор домена и классических подходов
docs/arena.md           роли, видимость, журнал, изоляция и smoke
docs/checkpoint-1.md    карта готовности к первому чекпоинту
docs/evolution.md       разбор эволюции (заполняется после кампании)
docs/solution.md        итоговое решение (заполняется после кампании)
docs/talk.md            план рассказа (заполняется к защите)
docs/adr/               решения по батчам и склейке повторов
case/                   кейс арены: данные, метрика, baseline, кампания
skill/                  навык: SKILL.md, scripts/, tests/ (этап ЧП3)
```

## Данные

CASTILLO (`sanitized`, ревизия `fb93f412e6bed3f3b834f0ded23e0865c286ed6c`, CC BY 4.0): 13 open-source моделей, 11 395 запросов из 7 источников. Train / public / private — 83 813 / 27 905 / 27 906 строк и 175 / 60 / 59 батчей. Строки с одинаковым `sample_id` склеены; split сделан по семействам похожих запросов, пересечений семейств между разделами нет.

Подробности и обоснование размеров — в [`case/dataset/README.md`](case/dataset/README.md), хэши и параметры — в `case/dataset/manifest.json` и `case/dataset/batches.json`. Public и private не коммитятся: они восстанавливаются генератором и остаются у оператора.

## Как воспроизвести

```bash
# 1. Данные: скачивание CASTILLO, склейка повторов, split по семействам
python3 case/dataset/generate.py
python3 case/dataset/batch.py

# 2. Проверка метрики, baseline и целостности данных (86 passed)
python3 -m pytest case/tests case/dataset/tests -q

# 3. Кейс копируется в arena-main/cases/ar09, затем подготовка и поиск
#    выполняются по prompts/prepare.md и prompts/start.md арены
```

Проверки кейса в арене: `uv run pytest cases/ar09/tests cases/ar09/dataset/tests -q` — 92 passed; `uv run pyright -p cases/ar09/pyrightconfig.json` — 0 errors. Полные свидетельства подготовки — в [`case/campaign/preparation.md`](case/campaign/preparation.md).

## Модели

Исследователи кампании: Claude Opus 5.5 (`claude-opus-5-5`) в Claude Code. Решатель и метрика LLM не используют. Модель-учитель и внешние модели в оценке не участвуют.
