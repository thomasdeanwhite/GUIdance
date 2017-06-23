import os
import subprocess


data_location = 'C:/work/leapmotion-study/data'
folders = ['ASLDigits/bin', 'GorogoaPuzzle/run', 'LeapmotionPaint', 'PolyDrop', 'VirtualPianoForBeginners/App']
frame_gens = ['RANDOM', 'STATE_DEPENDENT', 'SINGLE_MODEL', 'VQ', 'RANDOM']
iterations = 10
runtime = 60000
root = 'C:/Apps'

def run_app(folder):
  print(folder)

  #for frame_gen in frame_gens:
  #  run_single_config(folder, frame_gen, runtime)

  run_single_config(folder, 'RAW_RECONSTRUCTION', runtime)
  #run_single_config(folder, 'USER_PLAYBACK', runtime)

  output = folder.split('/')[0]

  return

def run_single_config(folder, frame_gen, runtime):

  options = '-frameSelectionStrategy ' + frame_gen
  run_one(folder, options, runtime)
  return

def run_one(folder, options, runtime):
  app_home = root + '/' + folder
  runpath = app_home + '/run.bat'
  os.chdir(app_home + '/NuiMimic/cfg')
  os.system('echo ' + options + ' ' + '-runtime ' + str(runtime) + '>testing-options.cfg')
  os.chdir(app_home + '/NuiMimic')
  os.system('rm current_run.nmDump')
  os.chdir(app_home)
  return_code = -1
  while return_code != 0:
    return_code = int(os.system('run.bat 1>log.txt 2>&1'))
    print(return_code)


for i in range(1, iterations):
  for folder in folders:
    run_app(folder)

