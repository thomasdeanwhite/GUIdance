# GuiWidgets #
Training models to identify GUI Widgets.

# Installation #

- clone repo
- Fetch weights using GitLFS: `git lfs pull` or download them from this repo
- Rename gui-tester/src/main/python/gui_interaction/config.py.BACK to config.py
- Change gui-tester/src/main/python/gui_interaction/config.py
  -  data_dir and output_dir to directory of training input data and desired output directory

# Running #

After downloading or training the model's  weights, run
```
gui-tester/src/main/python/gui_interaction/model_run.py [path to image]
```
to print out the identified bounding boxes.

To visualise bounding boxes, model_plot.py can be used:
```
gui-tester/src/main/python/gui_interaction/model_plot.py [path to image]
```
e.g.
```
python gui-tester/src/main/python/gui_interaction/model_plot.py /home/user/img.png
```

Ensure that the model weights are in the folder _weights_ in the directory you are running the python script from.
