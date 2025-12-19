@echo off
REM Train all 4 models with IMDb dataset for deep learning project
REM Each model will train for 15 epochs
REM Results will be saved to: results/deep_learning/imdb/

echo ========================================
echo Starting Training - All 4 Models (IMDb)
echo Dataset: IMDb 50K reviews
echo Epochs: 15 per model
echo Output: results/deep_learning/imdb/
echo ========================================

set DATA_PATH=data/processed/imdb_reviews.parquet
set EPOCHS=15
set BASE_OUTPUT=results/deep_learning/imdb

echo.
echo [1/4] Training Baseline LSTM...
echo Started at: %TIME%
python src/02_lstm_word2vec.py --data %DATA_PATH% --output_dir %BASE_OUTPUT%/01_lstm --epochs %EPOCHS%
echo Finished at: %TIME%

echo.
echo [2/4] Training Bi-LSTM...
echo Started at: %TIME%
python src/03_bilstm.py --data %DATA_PATH% --output_dir %BASE_OUTPUT%/02_bilstm --epochs %EPOCHS%
echo Finished at: %TIME%

echo.
echo [3/4] Training LSTM + Attention...
echo Started at: %TIME%
python src/04_lstm_attention.py --data %DATA_PATH% --output_dir %BASE_OUTPUT%/03_lstm_attention --epochs %EPOCHS%
echo Finished at: %TIME%

echo.
echo [4/4] Training GRU...
echo Started at: %TIME%
python src/05_gru.py --data %DATA_PATH% --output_dir %BASE_OUTPUT%/04_gru --epochs %EPOCHS%
echo Finished at: %TIME%

echo.
echo ========================================
echo All models training completed!
echo Results saved in: %BASE_OUTPUT%/
echo ========================================
pause
