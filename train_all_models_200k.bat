@echo off
REM Train all 4 models with 200K dataset for deep learning project
REM Each model will train for 5 epochs (~1 hour each = 4 hours total)

echo ========================================
echo Starting Training - All 4 Models
echo Dataset: 200K reviews
echo Epochs: 5 per model
echo Estimated total time: 4 hours
echo ========================================

set DATA_PATH=data/deep_learning_data/movies_reviews_200k.parquet
set EPOCHS=5

echo.
echo [1/4] Training Baseline LSTM...
echo Started at: %TIME%
wsl -d Ubuntu -u radian_try bash -c "cd '/mnt/c/Users/Radian Try/Documents/2nd Asia University (TW)/2nd Semester/Data Science/Midterm/movies-sentiment-starter' && python3 src/02_lstm_word2vec.py --data %DATA_PATH% --output_dir results/deep_learning/01_lstm --epochs %EPOCHS%"
echo Finished at: %TIME%

echo.
echo [2/4] Training Bi-LSTM...
echo Started at: %TIME%
wsl -d Ubuntu -u radian_try bash -c "cd '/mnt/c/Users/Radian Try/Documents/2nd Asia University (TW)/2nd Semester/Data Science/Midterm/movies-sentiment-starter' && python3 src/03_bilstm.py --data %DATA_PATH% --output_dir results/deep_learning/02_bilstm --epochs %EPOCHS%"
echo Finished at: %TIME%

echo.
echo [3/4] Training LSTM + Attention...
echo Started at: %TIME%
wsl -d Ubuntu -u radian_try bash -c "cd '/mnt/c/Users/Radian Try/Documents/2nd Asia University (TW)/2nd Semester/Data Science/Midterm/movies-sentiment-starter' && python3 src/04_lstm_attention.py --data %DATA_PATH% --output_dir results/deep_learning/03_lstm_attention --epochs %EPOCHS%"
echo Finished at: %TIME%

echo.
echo [4/4] Training GRU...
echo Started at: %TIME%
wsl -d Ubuntu -u radian_try bash -c "cd '/mnt/c/Users/Radian Try/Documents/2nd Asia University (TW)/2nd Semester/Data Science/Midterm/movies-sentiment-starter' && python3 src/05_gru.py --data %DATA_PATH% --output_dir results/deep_learning/04_gru --epochs %EPOCHS%"
echo Finished at: %TIME%

echo.
echo ========================================
echo All models training completed!
echo Results saved in: results/deep_learning/
echo ========================================
pause
