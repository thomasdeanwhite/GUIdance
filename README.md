# NuiMimic #
[![Build Status](https://travis-ci.org/thomasdeanwhite/NuiMimic.svg?branch=master)](https://travis-ci.org/thomasdeanwhite/NuiMimic)[![Coverage Status](https://coveralls.io/repos/github/thomasdeanwhite/NuiMimic/badge.svg?branch=master)](https://coveralls.io/github/thomasdeanwhite/NuiMimic?branch=master)

Repository for the NuiMimic automated software testing tool. NuiMimic is a tool which allows automated testing of applications which use Natural User Interfaces. Currently, only the Leap Motion Leap is supported.

![Image of Leapmotion Hands](http://i.imgur.com/pAg83Km.png)

# Installation #

To install:
- clone repo:
  * *git clone git@github.com:thomasdeanwhite/NuiMimic.git*
- install LeapJava.jar as local maven dependancy:
  * *cd [path to LeapJava.jar]*
  * *mvn install:install-file -Dfile=LeapJava.jar -DgroupId=com.leapmotion -DartifactId=leapmotion.sdk -Dpackaging=jar -Dversion=1.0*
- install dependencies
  * *cd [path to NuiMimic]*
  * *mvn clean install*
  
# Building #
To build, we use Maven Package:
- *cd [path to NuiMimic]*
- *mvn package*
- packaged jar is now in leap/target

# User Guide #

There are various steps to using NuiMimic:
1. Recording serialised Leap Motion frame data

2. Breaking the serialised frame data into raw NuiMimic data

3. Recording Screen State information

4. Processing NuiMimic data into final models.

# Runtime Options
| Key | Description |
| --- | --- |
| **Parameter Tuning** |  |
| Tmin:[arg]  | _Min value to tune (inclusive)_ |
| Tmax:[arg]  | _Max value to tune (exclusive)_ |
| Tparameter:[arg]  | _Parameter to tune_ |
| **Data Interpolation** |  |
| switchTime:[arg]  | _Time for interpolation between frames_ |
| bezierPoints:[arg]  | _Amount of points to use for Bezier Interpolation_ |
| **Leap Motion Sampling** |  |
| processPlayback  | _Should frames be processed during playback?_ |
| processScreenshots  | _Should screenshots be processed  during playback?_ |
| singleDataPool  | _Should a single data pool be used to reconstruct hands?_ |
| **Instrumentation** |  |
| untrackedPackages:[arg]  | _Packages to not be tracked when outputting lines and branches (comma separated)_ |
| sliceRoot:[arg]  | _Root for static slice through system_ |
| log_filename:[arg]  | _Select the file name for the log file. Files are divided into folders for coverage etc_ |
| instrumentation_approach:[arg]  | _Determines the approach to be used during class instrumentation. A static approach inserts calls to ClassAnalyzer.lineFound etc to track which lines/branches have been covered. Using an array stores all line/branch executions in an array of integers and has a method to get all the values_ |
| instrument_lines:[arg]  | _Switch on line instrumentation_ |
| instrument_branches:[arg]  | _Switch on branch instrumentation_ |
| write_class:[arg]  | _flag to determine whether or not to write classes. If set to true, the InstrumentingClassLoader will write out all classes to the value of BYTECODE_DIR_ |
| bytecode_dir:[arg]  | _directory in which to store bytecode if the WRITE_CLASS property is set to true_ |
| log_dir:[arg]  | _directory in which to store log files (application.log, timings.log)_ |
| log_timings:[arg]  | _set whether application timings should be written to a log file_ |
| use_changed_flag:[arg]  | _It is possible to add a flag through instrumentation that will tell the ClassAnalyzer that a class has changed in some way. This creates a form of hybrid approach to instrumentation, but saves work at the time of collecting coverage data_ |
| track_active_testcase:[arg]  | _When collecting coverage information, it is possible to include information about which test case covered each line. If this argument is true, use ClassAnalyzer.setActiveTest(TestCase), and then each line/branch object will have a list of test cases that cover it, accessed by CoverableGoal.getCoveringTests_ |
| **Experiments** |  |
| currentRun:[arg]  | _Can be used for experiments to output the current run (-1 will set to system runtime)_ |
| **Common** |  |
| runtype:[arg]  | _Type of run (default instrument)_ |
| **State Recognition** |  |
| histogramBins:[arg]  | _Amount of bins to sort pixels into for histogram comparison during generation guidence_ |
| histogramThreshold:[arg]  | _Difference required for two histograms to be considered unique states during generation guidence_ |
| screenshotCompression:[arg]  | _Order of magnitude to compress screenshots_ |
| **Output** |  |
| outputDir:[arg]  | _Directory for Output (default NuiMimic)_ |
| outputNullValue:[arg]  | _Output Value of Null Values ("NONE" by default)_ |
| outputExcludes:[arg]  | _Output options to exclude when logging_ |
| outputIncludes:[arg]  | _Output options to include when logging_ |
| **Data Processing** |  |
| clusters:[arg]  | _Amount of clusters to use for data processing_ |
| **Leap Motion Testing** |  |
| dataPoolDirectory:[arg]  | _Directory containing data pool_ |
| remainingBudget:[arg]  | _Remaining Budget after resuming from system halt_ |
| singleThread  | _Should frames be seeded on same thread as generation occurs?_ |
| showOutput  | _Should output be shown?_ |
| progress  | _Should progress be shown?_ |
| playbackFile:[arg]  | _File to playback (containing serialized ArrayList<com.leap.leapmotion.Frame> objects)_ |
| resumingFile:[arg]  | _CURRENT_RUN of run to resume after system halt or premature exit_ |
| framesPerSecond:[arg]  | _Number of frames to seed per second_ |
| startDelayTime:[arg]  | _Delay Time before frames are seeded_ |
| maxLoadedFrames:[arg]  | _Frames to retain for com.leap.leapmotion.Frame.frame(int [0->maxLoadedFrames]) method_ |
| runtime:[arg]  | _Time for testing application before exiting_ |
| input:[arg]  | _semicolon (;) separated list of files for input_ |
| visualiseData  | _Displays the currently seeded data in a separate window._ |
| invertZAxis  | _Inverts the direction the hand is facing_ |
| jitter:[arg]  | _Random amount to move all joints per frame_ |
| skipDependencyTree  | _Skip building dependency tree_ |
| dependencyTreeOverride  | _Use to always build a fresh dependency tree_ |
| **Oracle** |  |
| ThistogramThreshold:[arg]  | _Difference required for two histograms to be considered unique states for oracle_ |
| ThistogramBins:[arg]  | _Amount of bins to sort pixels into for histogram comparison for oracle_ |
| **Web Testing** |  |
| webpage:[arg]  | _Webpage containing Leap Motion app to test_ |
| **Leap Motion Gestures** |  |
| gestureCircleMinRadius:[arg]  | _Minimum radius a circle gesture can be_ |
| gestureCircleCentreFrames:[arg]  | _Number of previous frames used to calculated a circle gesture._ |
| gestureTimeLimit:[arg]  | _Duration to seed gestures for_ |
| **Leap Motion Instrumentation** |  |
| jar:[arg]  | _Jar to instrument_ |
| excludedPackages:[arg]  | _Additional packages to exclude from instrumentation_ |
| forbiddenPackages:[arg]  | _Override packages to exclude from instrumentation_ |
| cp:[arg]  | _Path to library files for application_ |
| replace_fingers_method  | _Replaces com.leap.leapmotion.FingerList.fingers() method with com.leap.leapmotion.FingerList.extended() [for older API versions]_ |
| recording  | _Records Leap Motion data to storage_ |
| controllerSuperClass  | _The Controller class is extended instead of instantiated_ |
| frameSelectionStrategy:[arg]  | _Strategy for Frame Selection_ |
