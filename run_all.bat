@echo off
setlocal

rem === 引数: ポート番号（省略可） ===
set PORT=%1
if "%PORT%"=="" set PORT=8787
if not "%1"=="" shift

rem === Python 実行ファイルの自動検出（venv優先） ===
set PYTHON=python
if exist ".venv311\Scripts\python.exe" set PYTHON=.venv311\Scripts\python.exe

rem === 環境変数で runner のURL上書き ===
set GEN_URL=http://127.0.0.1:%PORT%/gen/generate_4
set TTS_URL=http://127.0.0.1:%PORT%/tts/speak
set TTS_TIMEOUT_SEC=60

echo [info] Using port %PORT%
echo [info] GEN_URL=%GEN_URL%
echo [info] TTS_URL=%TTS_URL%

rem === すでにLISTEN中か確認（別ウィンドウでサーバ起動しないように） ===
set PID=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set PID=%%p

if not defined PID (
  echo [info] starting server...
  start "AI4nin Server %PORT%" cmd /c "%PYTHON% -m uvicorn app.server_all:app --host 127.0.0.1 --port %PORT%"
) else (
  echo [info] server already listening on %PORT% (PID %PID%)
)

rem === ヘルスチェック（最大 ~20秒待つ） ===
powershell -NoProfile -Command ^
  "$u='http://127.0.0.1:%PORT%/health';" ^
  "for($i=0;$i -lt 40;$i++){" ^
  "  try{$r=Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 2;if($r.StatusCode -eq 200){exit 0}}" ^
  "  catch{}; Start-Sleep -Milliseconds 500}; exit 1"

if errorlevel 1 (
  echo [error] server not responding on port %PORT% だッピ
  exit /b 1
)

rem === runner 起動（残りの引数はそのまま runner へ渡す） ===
echo [run] launching talk_runner...
set GEN_URL=%GEN_URL%
set TTS_URL=%TTS_URL%
"%PYTHON%" app\talk_runner.py %*

endlocal
