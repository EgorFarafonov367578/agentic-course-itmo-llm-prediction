# Чекпоинт 1: готовность AR-09

Статус на 06.11.2026: **артефакты ЧП1 подготовлены, кампания ещё не запускалась.** В журнале арены 0 из 40 решений; это ожидаемо для чекпоинта, на котором нужны подготовленный кейс и прошедший smoke, а не результаты эволюции.

| Требование ЧП1 | Где подтверждение | Статус |
| --- | --- | --- |
| Тема согласована | `README.md`, `case/README.md`: тема AR-09 | Внешнее действие: отправленная форма/согласование с `kudep` в репозитории не хранится; подтвердить отдельно |
| Обзор домена | `docs/domain.md` | Готово |
| Простой baseline | `case/ways/baseline/`, `case/tests/test_baseline.py` | Готово: квантили `59 / 367 / 779` |
| Smoke арены пройден | `docs/arena.md`, `case/campaign/preparation.md` | Зафиксирован `passed`; тесты кейса и pyright повторно пройдены |
| Кейс оформлен | `case/README.md`, `case/campaign/campaign.md`, `case/metric/`, `case/dataset/` | Готово |
| Метрика с тестами | `case/metric/score.py`, `case/metric/aggregate.py`, `case/tests/test_metric.py` | Готово |
| Split обоснован | `case/dataset/README.md`, `case/dataset/manifest.json`, `case/dataset/batches.json` | Готово, проверяется `case/dataset/tests/test_data_integrity.py` |

## Фактические проверки

- Данные: 83 813 / 27 905 / 27 906 строк; 175 / 60 / 59 батчей; хэши файлов совпадают с manifest; id уникальны; семейства train/public/private не пересекаются.
- Baseline: `lo=59`, `pred=367`, `hi=779`; MAE на public `0.743`, полная метрика `1.4792` на public.
- Оракул по собственным ответам строки: MAE `0.152`, полная метрика около `0.282`.
- Метрика: `ar09-quantile-v1`, `minimize`, штраф `40` за невалидную строку; агрегация — по строкам, а не по батчам.
- Пакет `case/metric` проверен контрактом арены: `freeze_metric` и `load_metric` дают version `ar09-quantile-v1`, kind `command`, direction `minimize`, команды `score.py` и `aggregate.py`.
- Тесты кейса в арене: `92 passed`; pyright: `0 errors`. В студенческом репозитории: `86 passed`, `ruff check .` без замечаний.
- Полная сборка данных и нарезка батчей воспроизводятся побайтно; baseline end-to-end на public через `solve.py → score.py → aggregate.py` даёт `1.4792295985120796`.

## Что осталось до запуска поиска

1. Подключить public и private на тех же данных `castillo-batched-v2`, метрике `ar09-quantile-v1`, образе, лимитах и release `ar09-c1-v1`.
2. Выполнить авторизованный `search-readiness` private и записать ответ без токена.
3. Выдать исследователю public- и private-токены.
4. Запустить поиск отдельной командой по `case/campaign/start.md`.
