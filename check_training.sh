#!/bin/bash
echo "=== Статус обучения ==="
echo "Время: $(date '+%H:%M:%S')"
echo ""
echo "Последний чекпоинт:"
ls -lht checkpoints/*.ckpt 2>/dev/null | head -1
echo ""
echo "Количество чекпоинтов:"
ls -1 checkpoints/*.ckpt 2>/dev/null | wc -l
echo ""
echo "Размер чекпоинтов:"
du -sh checkpoints/ 2>/dev/null
