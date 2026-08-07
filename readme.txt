1. Set the data path in terminal (one-time)
"setx SKINSENSE_DATA_DIR "<path_to_project_folder>\Skin Cancer MNIST HAM10000\datasets""

2. project tree should be like this

"Skin Cancer MNIST HAM10000/
├── __pycache__/
│   └── preprocessing_optimized.cpython-310.pyc
├── datasets/
│   ├── HAM10000_images_part_1/
│   │   ├── ISIC_0024306.jpg
│   │   ├── ISIC_0024307.jpg
│   │   ├── ISIC_0024308.jpg
│   │   ├── ISIC_0024309.jpg
│   │   ├── ISIC_0024310.jpg
│   │   ├── ISIC_0024311.jpg
│   │   ├── ISIC_0024312.jpg
│   │   ├── ISIC_0024313.jpg
│   │   ├── ISIC_0024314.jpg
│   │   ├── ISIC_0024315.jpg
│   │   └── ...
│   ├── HAM10000_images_part_2/
│   │   ├── ISIC_0029306.jpg
│   │   ├── ISIC_0029307.jpg
│   │   ├── ISIC_0029308.jpg
│   │   ├── ISIC_0029309.jpg
│   │   ├── ISIC_0029310.jpg
│   │   ├── ISIC_0029311.jpg
│   │   ├── ISIC_0029312.jpg
│   │   ├── ISIC_0029313.jpg
│   │   ├── ISIC_0029314.jpg
│   │   ├── ISIC_0029315.jpg
│   │   └── ...
│   ├── hmnist_28_28_L.csv/
│   │   └── hmnist_28_28_L.csv
│   ├── hmnist_28_28_RGB.csv/
│   │   └── hmnist_28_28_RGB.csv
│   ├── hmnist_8_8_L.csv/
│   │   └── hmnist_8_8_L.csv
│   ├── hmnist_8_8_RGB.csv/
│   │   └── hmnist_8_8_RGB.csv
│   └── HAM10000_metadata.csv
├── outputs/  								<-- folder gets autocreated
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_1.pth
│   │   ├── checkpoint_epoch_14.pth
│   │   ├── checkpoint_epoch_23.pth
│   │   ├── checkpoint_epoch_27.pth
│   │   ├── checkpoint_epoch_3.pth
│   │   ├── checkpoint_epoch_4.pth
│   │   ├── checkpoint_epoch_9.pth
│   │   └── quantized_model.pth
│   ├── logs/
│   │   ├── confusion_matrix_float.png
│   │   ├── confusion_matrix_quantized.png
│   │   ├── training_curves.png
│   │   ├── training_history.json
│   │   ├── training_mobilenet_v3_2026-07-26_07-21PM.log
│   │   └── training_summary.json
│   └── splits/
│       ├── label_encoder.pkl
│       ├── test_split.csv
│       ├── train_split.csv
│       └── val_split.csv
├── skinsense_env/
│   ├── Include/
│   ├── Lib/
│   │   └── site-packages/
│   │       ├── __pycache__/
│   │       │   ├── isympy.cpython-310.pyc
│   │       │   ├── pylab.cpython-310.pyc
│   │       │   ├── readline.cpython-310.pyc
│   │       │   ├── six.cpython-310.pyc
│   │       │   ├── threadpoolctl.cpython-310.pyc
│   │       │   └── typing_extensions.cpython-310.pyc
│   │       ├── _distutils_hack/
│   │       │   ├── __pycache__/
│   │       │   │   ├── __init__.cpython-310.pyc
│   │       │   │   └── override.cpython-310.pyc
│   │       │   ├── __init__.py
│   │       │   └── override.py
│   │       ├── colorama/
│   │       │   ├── __pycache__/
│   │       │   │   ├── __init__.cpython-310.pyc
│   │       │   │   ├── ansi.cpython-310.pyc
│   │       │   │   ├── ansitowin32.cpython-310.pyc
│   │       │   │   ├── initialise.cpython-310.pyc
│   │       │   │   ├── win32.cpython-310.pyc
│   │       │   │   └── winterm.cpython-310.pyc
│   │       │   ├── tests/
│   │       │   │   ├── __pycache__/
│   │       │   │   │   ├── __init__.cpython-310.pyc
│   │       │   │   │   ├── ansi_test.cpython-310.pyc
│   │       │   │   │   ├── ansitowin32_test.cpython-310.pyc
│   │       │   │   │   ├── initialise_test.cpython-310.pyc
│   │       │   │   │   ├── isatty_test.cpython-310.pyc
│   │       │   │   │   ├── utils.cpython-310.pyc
│   │       │   │   │   └── winterm_test.cpython-310.pyc
│   │       │   │   ├── __init__.py
│   │       │   │   ├── ansi_test.py
│   │       │   │   ├── ansitowin32_test.py
│   │       │   │   ├── initialise_test.py
│   │       │   │   ├── isatty_test.py
│   │       │   │   ├── utils.py
│   │       │   │   └── winterm_test.py
│   │       │   ├── __init__.py
│   │       │   ├── ansi.py
│   │       │   ├── ansitowin32.py
│   │       │   ├── initialise.py
│   │       │   ├── win32.py
│   │       │   └── winterm.py
│   │       ├── colorama-0.4.6.dist-info/
│   │       │   ├── licenses/
│   │       │   │   └── LICENSE.txt
│   │       │   ├── INSTALLER
│   │       │   ├── METADATA
│   │       │   ├── RECORD
│   │       │   ├── REQUESTED
│   │       │   └── WHEEL
│   │       ├── coloredlogs/
│   │       │   ├── __pycache__/
│   │       │   │   ├── __init__.cpython-310.pyc
│   │       │   │   ├── cli.cpython-310.pyc
│   │       │   │   ├── demo.cpython-310.pyc
│   │       │   │   ├── syslog.cpython-310.pyc
│   │       │   │   └── tests.cpython-310.pyc
│   │       │   ├── converter/
│   │       │   │   ├── __pycache__/
│   │       │   │   │   ├── __init__.cpython-310.pyc
│   │       │   │   │   └── colors.cpython-310.pyc
│   │       │   │   ├── __init__.py
│   │       │   │   └── colors.py
│   │       │   ├── __init__.py
│   │       │   ├── cli.py
│   │       │   ├── demo.py
│   │       │   ├── syslog.py
│   │       │   └── tests.py
│   │       ├── coloredlogs-15.0.1.dist-info/
│   │       │   ├── entry_points.txt
│   │       │   ├── INSTALLER
│   │       │   ├── LICENSE.txt
│   │       │   ├── METADATA
│   │       │   ├── RECORD
│   │       │   ├── top_level.txt
│   │       │   └── WHEEL
│   │       ├── contourpy/
│   │       │   ├── __pycache__/
│   │       │   │   ├── __init__.cpython-310.pyc
│   │       │   │   ├── _version.cpython-310.pyc
│   │       │   │   ├── array.cpython-310.pyc
│   │       │   │   ├── chunk.cpython-310.pyc
│   │       │   │   ├── convert.cpython-310.pyc
│   │       │   │   ├── dechunk.cpython-310.pyc
│   │       │   │   ├── enum_util.cpython-310.pyc
│   │       │   │   ├── typecheck.cpython-310.pyc
│   │       │   │   └── types.cpython-310.pyc
│   │       │   ├── util/
│   │       │   │   ├── __pycache__/
│   │       │   │   │   ├── __init__.cpython-310.pyc
│   │       │   │   │   ├── _build_config.cpython-310.pyc
│   │       │   │   │   ├── bokeh_renderer.cpython-310.pyc
│   │       │   │   │   ├── bokeh_util.cpython-310.pyc
│   │       │   │   │   ├── data.cpython-310.pyc
│   │       │   │   │   ├── mpl_renderer.cpython-310.pyc
│   │       │   │   │   ├── mpl_util.cpython-310.pyc
│   │       │   │   │   └── renderer.cpython-310.pyc
│   │       │   │   ├── __init__.py
│   │       │   │   ├── _build_config.py
│   │       │   │   ├── bokeh_renderer.py
│   │       │   │   ├── bokeh_util.py
│   │       │   │   ├── data.py
│   │       │   │   ├── mpl_renderer.py
│   │       │   │   ├── mpl_util.py
│   │       │   │   └── renderer.py
│   │       │   ├── __init__.py
│   │       │   ├── _contourpy.cp310-win_amd64.lib
│   │       │   ├── _contourpy.cp310-win_amd64.pyd
│   │       │   ├── _contourpy.pyi
│   │       │   ├── _version.py
│   │       │   ├── array.py
│   │       │   ├── chunk.py
│   │       │   ├── convert.py
│   │       │   └── ...
│   │       ├── contourpy-1.3.2.dist-info/
│   │       │   ├── INSTALLER
│   │       │   ├── LICENSE
│   │       │   ├── METADATA
│   │       │   ├── RECORD
│   │       │   └── WHEEL
│   │       ├── cycler/
│   │       │   ├── __pycache__/
│   │       │   │   └── __init__.cpython-310.pyc
│   │       │   ├── __init__.py
│   │       │   └── py.typed
│   │       ├── cycler-0.12.1.dist-info/
│   │       │   ├── INSTALLER
│   │       │   ├── LICENSE
│   │       │   ├── METADATA
│   │       │   ├── RECORD
│   │       │   ├── top_level.txt
│   │       │   └── WHEEL
│   │       └── ...
│   ├── Scripts/
│   │   ├── activate
│   │   ├── activate.bat
│   │   ├── Activate.ps1
│   │   ├── backend-test-tools.exe
│   │   ├── check-model.exe
│   │   ├── check-node.exe
│   │   ├── coloredlogs.exe
│   │   ├── deactivate.bat
│   │   ├── f2py.exe
│   │   ├── fonttools.exe
│   │   └── ...
│   ├── share/
│   │   └── man/
│   │       └── man1/
│   │           ├── isympy.1
│   │           └── ttx.1
│   └── pyvenv.cfg
├── export_for_rpi.py
├── preprocessing_optimized.py
├── readjusted_train.py
├── readme.txt
├── requirements.txt
├── rpi_inference.py
└── ..."

3. Create and activate the virtual environment
"python -m venv skinsense_env
skinsense_env\Scripts\activate"

4. Install dependencies
"pip install -r requirements.txt"

5. Verify CUDA is visible (optional)
"python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"" <-- should print "True NVIDIA GeForce <gpu_model>"

6. Run training
"python readjusted_train.py"

7. Check the results
outputs/logs/training_curves.png — loss/accuracy curves
outputs/logs/training_history.json — raw numbers per epoch
outputs/checkpoints/best_model.pth — the best model weights
Console/log will print final train vs. quantized test accuracy

8. Export for Raspberry Pi (after training finishes)
"python export_for_rpi.py"

9. Deploy to the Pi
Copy the entire outputs/rpi_export/ folder to the Raspberry Pi, along with rpi_inference.py and some test images.
On the Pi:
"python3 -m venv skinsense_env
source skinsense_env/bin/activate
pip install onnxruntime pillow numpy scikit-learn
python3 rpi_inference.py test_images"