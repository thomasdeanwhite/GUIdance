# GuiWidgets #
Training models to identify GUI Widgets.

# Installation #

- clone repo
- Rename gui-tester/src/main/python/gui_interaction/config.py.BACK to config.py
- Change gui-tester/src/main/python/gui_interaction/config.py
  -  data_dir and output_dir to directory of training input data and desired output directory

# Running #

After downloading or training a model and creating weights, run
```
gui-tester/src/main/python/gui_interaction/model_plot.py [path to image]
```

e.g.

```
python gui-tester/src/main/python/gui_interaction/model_plot.py /home/user/img.png
```

Ensure that the model weights are in the folder _weights_ in the directory you are running the python script from.