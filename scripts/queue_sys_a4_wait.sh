#!/usr/bin/env bash
until grep -q "\[done\] sys_a2" /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/logs/sys_presence.log 2>/dev/null; do sleep 60; done
exec /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/scripts/queue_sys_a4.sh 1
