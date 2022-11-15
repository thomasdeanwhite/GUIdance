# GuiWidgets #
A technique to identify widgets in screenshots of GUIs. 

# Installation #
- `git clone git@github.com:thomasdeanwhite/GUIdance.git`
- Fetch weights using GitLFS: `git lfs pull` or download them from this repo 
- unzip weights zip file
- Copy `gui-tester/src/main/python/gui_interaction/config.py.BACK` to `config.py`
- Install dependencies
  - `pip3 install -r requirements.txt`
- Change `gui-tester/src/main/python/gui_interaction/config.py`
  -  `data_dir` and `output_dir` to directory of training input data and desired output directory

# Running #

To visualise bounding boxes, model_plot.py can be used:
```
gui-tester/src/main/python/gui_interaction/model_plot.py [path to image]
```
e.g.
```
python gui-tester/src/main/python/gui_interaction/model_plot.py /home/user/img.png
```


![Annotated App](https://raw.githubusercontent.com/thomasdeanwhite/GUIdance/master/public/app-annotated.png)

Ensure that the model weights are in the folder _weights_ in the directory you are running the python script from.

The weights file can be found here: https://github.com/thomasdeanwhite/GUIdance/issues/7
