# CUDA Setup Guide for TensorFlow 2.20 + RTX 4060

## Step 1: Install CUDA Toolkit 12.6

1. Download from: https://developer.nvidia.com/cuda-downloads
   - Select: Windows > x86_64 > 11 > exe (local)
   - File size: ~3.5 GB

2. Run the installer
   - Choose "Express Installation" (recommended)
   - Wait for installation (~10-15 minutes)

3. Verify installation:
   ```bash
   nvcc --version
   ```

## Step 2: Install cuDNN 9.x

1. Go to: https://developer.nvidia.com/cudnn
   - Click "Download cuDNN"
   - Sign in/Create NVIDIA Developer account (free)

2. Download: cuDNN v9.x for CUDA 12.x (Windows)
   - File: cudnn-windows-x86_64-9.x.x.xx_cuda12-archive.zip (~800 MB)

3. Extract the ZIP file

4. Copy files to CUDA directory:
   ```
   Copy from extracted folder TO CUDA installation:

   bin\cudnn*.dll        → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\
   include\cudnn*.h      → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\
   lib\x64\cudnn*.lib    → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\
   ```

## Step 3: Add to System PATH (if not already added)

1. Press `Win + X` → System
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", find "Path", click "Edit"
5. Verify these paths exist (installer usually adds them):
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp
   ```

## Step 4: Restart Computer

After installing CUDA and cuDNN, **restart your computer** for changes to take effect.

## Step 5: Test GPU Detection

After restart, run:
```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Troubleshooting

If GPU not detected:
1. Make sure NVIDIA driver is up to date
2. Verify CUDA installation: `nvcc --version`
3. Check PATH environment variables
4. Restart computer again

## Notes

- TensorFlow 2.20 works with CUDA 12.x
- RTX 4060 requires CUDA 12.x or later
- Total download size: ~4-5 GB
- Installation time: ~30-40 minutes
