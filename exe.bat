@echo off
cd /d "C:\Users\julio\Desktop\codes\trade3\framework"
call venv\Scripts\activate
python run_otimizador.py --config configs/bollinger_stoch_eurusd.yaml --dashboard
pause