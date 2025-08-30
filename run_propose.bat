@echo off
setlocal

rem === 引数: ポート番号（省略可） ===
set PORT=%1
if "%PORT%"=="" set PORT=8787
if not "%1"=="" shift

rem === Python 実行ファイルの自動検出（venv優先） ===
set PYTHON=python
if exist ".venv311\Scripts\python.exe" set PYTHON=.venv311\Scripts\python.exe

rem === 環境変数: propose ループ用 ===
set GEN_URL=http://127.0.0.1:%PORT%/gen/generate_4
set TTS_URL=http://127.0.0.1:%PORT%/tts/speak
set PAR_URL=http://127.0.0.1:%PORT%/gen/paratalk
set PROPOSE_URL=http://127.0.0.1:%PORT%/gen/propose
set PROPOSE_STEPS=10
set TTS_TIMEOUT_SEC=60

echo [info] Using port %PORT%
echo [info] PROPOSE_URL=%PROPOSE_URL%
echo [info] PROPOSE_STEPS=%PROPOSE_STEPS%

rem === サーバ起動確認 ===
set PID=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set PID=%%p

if not defined PID (
  echo [info] starting server...
  start "AI4nin Server %PORT%" cmd /c "%PYTHON% -m uvicorn app.server_all:app --host 127.0.0.1 --port %PORT%"
) else (
  echo [info] server already listening on %PORT% (PID %PID%)
)

rem === ヘルスチェック ===
powershell -NoProfile -Command ^
  "$u='http://127.0.0.1:%PORT%/health';" ^
  "for($i=0;$i -lt 40;$i++){" ^
  "  try{$r=Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 2;if($r.StatusCode -eq 200){exit 0}}" ^
  "  catch{}; Start-Sleep -Milliseconds 500}; exit 1"

if errorlevel 1 (
  echo [error] server not responding on port %PORT%
  exit /b 1
)

rem === talk_runner を propose ループモードで起動 ===
echo [run] launching talk_runner (propose loop)...
"%PYTHON%" -m app.talk_runner %*

endlocal
