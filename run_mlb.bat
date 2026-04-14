@echo off
cd /d C:\MLB
set PYTHONIOENCODING=utf-8
C:\Users\codys\AppData\Local\Programs\Python\Python312\python.exe data\mlb.py >> C:\MLB\mlb_log.txt 2>&1