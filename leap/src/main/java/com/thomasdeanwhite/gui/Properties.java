package com.thomasdeanwhite.gui;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.sheffield.instrumenter.InstrumentationProperties;
import com.sheffield.instrumenter.analysis.ClassAnalyzer;
import com.sheffield.instrumenter.instrumentation.objectrepresentation.BranchHit;
import com.sheffield.instrumenter.instrumentation.objectrepresentation.LineHit;
import com.thomasdeanwhite.gui.util.ClassTracker;
import com.thomasdeanwhite.gui.util.FileHandler;
import com.sheffield.output.Csv;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.UnrecognizedOptionException;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

/**
 * Created by thomas on 04/05/2016.
 */
public class Properties extends InstrumentationProperties {


    @Parameter(key = "N", description = "N-Gram Length to use for data processing", category = "Data Processing")
    public static int N = 3;

    public static final boolean SHOW_DISPLAY = false;

    @Parameter(key = "clusters", description = "Amount of clusters to use for data processing", category = "Data Processing")
    public static int CLUSTERS = 400;

    public enum Interaction {
        USER, MONKEY, DEEP_LEARNING
    }

    @Parameter(key = "interaction", description = "Technique for application interaction", hasArgs = true, category = "GUI Testing")
    public static Interaction INTERACTION = Interaction.USER;

    /*
                    Properties for Leap Motion Testing
             */
    @Parameter(key = "dataPoolDirectory", description = "Directory containing data pool", hasArgs = true, category = "Leap Motion Testing")
    public static String DIRECTORY = System.getProperty("user.home") + "/data/leap-motion";

    @Parameter(key = "remainingBudget", description = "Remaining Budget after resuming from system halt", hasArgs = true, category = "Leap Motion Testing")
    public static long REMAINING_BUDGET = -1;

    @Parameter(key = "singleThread", description = "Should frames be seeded on same thread as generation occurs?", hasArgs = false, category = "Leap Motion Testing")
    public static boolean SINGLE_THREAD = true;

    @Parameter(key = "showOutput", description = "Should output be shown?", hasArgs = false, category = "Leap Motion Testing")
    public static boolean SHOW_OUTPUT = true;

    @Parameter(key = "progress", description = "Should progress be shown?", hasArgs = false, category = "Leap Motion Testing")
    public static boolean SHOW_PROGRESS = false;

    @Parameter(key = "webpage", description = "Webpage containing Leap Motion app to test", category = "Web Testing")
    public static String WEBSITE = null;


    @Parameter(key = "playbackFile", description = "File to playback (containing serialized ArrayList<com.leap.gui.Frame> objects)", hasArgs = true, category = "Leap Motion Testing")
    public static String PLAYBACK_FILE = null;

    @Parameter(key = "resumingFile", description = "CURRENT_RUN of run to resume after system halt or premature exit", hasArgs = true, category = "Leap Motion Testing")
    public static int RESUME_RUN = -1;

    @Parameter(key = "framesPerSecond", description = "Number of frames to seed per second", hasArgs = true, category = "Leap Motion Testing")
    public static long FRAMES_PER_SECOND = 200;

    @Parameter(key = "switchTime", description = "Time for interpolation between frames", hasArgs = true, category = "Data Interpolation")
    public static int SWITCH_TIME = 5;//20fps

    @Parameter(key = "bezierPoints", description = "Amount of points to use for Bezier Interpolation", hasArgs = true, category = "Data Interpolation")
    public static int BEZIER_POINTS = 1;

    @Parameter(key = "startDelayTime", description = "Delay Time before frames are seeded", hasArgs = true, category = "Leap Motion Testing")
    public static long DELAY_TIME = 15000;

    @Parameter(key = "maxLoadedFrames", description = "Frames to retain for com.leap.gui.Frame.frame(int [0->maxLoadedFrames]) method", hasArgs = true, category = "Leap Motion Testing")
    public static int MAX_LOADED_FRAMES = 200;

    @Parameter(key = "runtime", description = "Time for testing application before exiting", hasArgs = true, category = "Leap Motion Testing")
    public static long RUNTIME = 600000;

    @Parameter(key = "currentRun", description = "Can be used for experiments to output the current run (-1 will set to system runtime)", hasArgs = true, category = "Experiments")
    public static long CURRENT_RUN = -1;

    @Parameter(key = "gestureCircleMinRadius", description = "Minimum radius a circle gesture can be", hasArgs = true, category = "Leap Motion Gestures")
    public static int GESTURE_CIRCLE_RADIUS = 5;

    @Parameter(key = "gestureCircleCentreFrames", description = "Number of previous frames used to calculated a circle gesture.", hasArgs = true, category = "Leap Motion Gestures")
    public static int GESTURE_CIRCLE_FRAMES = 0;

    @Parameter(key = "gestureTimeLimit", description = "Duration to seed gestures for", hasArgs = true, category = "Leap Motion Gestures")
    public static int GESTURE_TIME_LIMIT = 500;


    @Parameter(key = "untrackedPackages", description = "Packages to not be tracked when outputting lines and branches (comma separated)", hasArgs = true, category = "Instrumentation")
    public static String UNTRACKED_PACKAGES = null;

    @Parameter(key = "sliceRoot", description = "Root for static slice through system", hasArgs = true, category = "Instrumentation")
    public static String SLICE_ROOT = null;

    @Parameter(key = "input", description = "semicolon (;) separated list of files for input", hasArgs = true, category = "Leap Motion Testing")
    public static String INPUT_STRING = null;
    public static String[] INPUT = {"default-user"}; //derived from GESTURE_FILE_STRING

    @Parameter(key = "visualiseData", description = "Displays the currently seeded data in a separate window.", hasArgs = false, category = "Leap Motion Testing")
    public static boolean VISUALISE_DATA = false;


    /*
        Properties for Leap Motion Instrumentation
    */
    @Parameter(key = "jar", description = "Jar to instrument", hasArgs = true, category = "Leap Motion Instrumentation")
    public static String JAR_UNDER_TEST = null;


    /*
    Properties for Leap Motion Instrumentation
*/
    @Parameter(key = "exec", description = "Command to start AUT", hasArgs = true, category = "Testing")
    public static String EXEC = null;

    @Parameter(key = "excludedPackages", description = "Additional packages to exclude from instrumentation", hasArgs = true, category = "Leap Motion Instrumentation")
    public static String EXCLUDED_PACKAGES_STRING = null;

    @Parameter(key = "forbiddenPackages", description = "Override packages to exclude from instrumentation", hasArgs = true, category = "Leap Motion Instrumentation")
    public static String FORBIDDEN_PACKAGES_STRING = null;

    public static String[] FORBIDDEN_PACKAGES = new
            String[]{"com/sheffield/leapmotion/",
            "com/google/", "org/xml",
            "com/gui/", "java/", "org/json/", "org/apache/commons/cli/",
            "org/junit/", "org/apache", "com/garg", "net/sourceforge",
            "com/steady", "com/thought", "com/jogamp", "com/bulletphysics", "com/jme3",
            "com/oracle", "org/objectweb", "javax", "jogamp", "jme3", "org/lwjgl", "net/java"};

    @Parameter(key = "cp", description = "Path to library files for application", hasArgs = true, category = "Leap Motion Instrumentation")
    public static String CLASS_PATH = "./lib";

    @Parameter(key = "replace_fingers_method", description = "Replaces com.leap.gui.FingerList.fingers() method with com.leap.gui.FingerList.extended() [for older API versions]", hasArgs = false, category = "Leap Motion Instrumentation")
    public static boolean REPLACE_FINGERS_METHOD = false;

    @Parameter(key = "recording", description = "Records Leap Motion data to storage", hasArgs = false, category = "Leap Motion Instrumentation")
    public static boolean RECORDING = false;

    @Parameter(key = "controllerSuperClass", description = "The Controller class is extended instead of instantiated", hasArgs = false, category = "Leap Motion Instrumentation")
    public static boolean CONTROLLER_SUPER_CLASS = false;

    @Parameter(key = "invertZAxis", description = "Inverts the direction the hand is facing", hasArgs = false, category = "Leap Motion Testing")
    public static boolean INVERT_Z_AXIS = false;

    @Parameter(key = "ignoreDictionary", description = "Ignores previous run data", hasArgs = false, category = "Deep Learning")
    public static boolean IGNORE_DICTIONARY = false;


    public enum FrameSelectionStrategy {
        RANDOM, EMPTY, VQ, STATE_DEPENDENT, SINGLE_MODEL, RECONSTRUCTION, RAW_RECONSTRUCTION, REGRESSION, NONE, MANUAL, STATE_ISOLATED, USER_PLAYBACK
    }

    @Parameter(key = "frameSelectionStrategy", description = "Strategy for Frame Selection", hasArgs = true, category = "Leap Motion Instrumentation")
    public static FrameSelectionStrategy FRAME_SELECTION_STRATEGY = FrameSelectionStrategy.STATE_DEPENDENT;

    @Parameter(key = "histogramBins", description = "Amount of bins to sort pixels into for histogram comparison during generation guidence", hasArgs = true,
            category =
            "State " +
            "Recognition")
    public static int HISTOGRAM_BINS = 25;

    @Parameter(key = "histogramThreshold", description = "Difference required for two histograms to be considered unique states during generation guidence", hasArgs =
            true,
            category = "State Recognition")
    public static float HISTOGRAM_THRESHOLD = 0.1f;

    @Parameter(key = "ThistogramThreshold", description = "Difference required for two histograms to be considered unique states for oracle", hasArgs = true, category =
            "Oracle")
    public static float TESTING_HISTOGRAM_THRESHOLD = 0.003f;

    @Parameter(key = "ThistogramBins", description = "Amount of bins to sort pixels into for histogram comparison for oracle", hasArgs = true, category = "Oracle")
    public static int TESTING_HISTOGRAM_BINS = 25;

    @Parameter(key = "screenshotCompression", description = "Order of magnitude to compress screenshots", hasArgs = true, category = "State Recognition")
    public static int SCREENSHOT_COMPRESSION = 4;

    @Parameter(key = "jitter", description = "Random amount to move all joints per frame", hasArgs = true, category = "Leap Motion Testing")
    public static float JITTER = 0f;

    /*
     * Output formatting properties
     */

    @Parameter(key = "outputDir", description = "Directory for Output (default NuiMimic)", hasArgs = true, category = "Output")
    public static String TESTING_OUTPUT = "NuiMimic";

    @Parameter(key = "outputNullValue", description = "Output Value of Null Values (\"NONE\" by default)", hasArgs = true, category = "Output")
    public static String NULL_VALUE_OUTPUT = "NONE";

    @Parameter(key = "outputExcludes", description = "Output options to exclude when logging", hasArgs = true, category = "Output")
    public static String OUTPUT_EXCLUDES = "outputNullValue,outputExcludes,jar,cp,leave_leapmotion_alone,replace_fingers_method,Tmin,Tmax,Tparameter,outputDir";

    public static ArrayList<String> OUTPUT_EXCLUDES_ARRAY;

    @Parameter(key = "outputIncludes", description = "Output options to include when logging", hasArgs = true, category = "Output")
    public static String OUTPUT_INCLUDES;

    public static ArrayList<String> OUTPUT_INCLUDES_ARRAY;


    /*
     * Properties for tuning parameters
     */
    @Parameter(key = "Tmin", description = "Min value to tune (inclusive)", hasArgs = true, category = "Parameter Tuning")
    public static float MIN_TUNING_VALUE = 0f;

    @Parameter(key = "Tmax", description = "Max value to tune (exclusive)", hasArgs = true, category = "Parameter Tuning")
    public static float MAX_TUNING_VALUE = 1f;

    @Parameter(key = "Tparameter", description = "Parameter to tune", hasArgs = true, category = "Parameter Tuning")
    public static String TUNING_PARAMETER = null;


    public enum RunType {
        INSTRUMENT, VISUALISE, RECONSTRUCT, STATE_RECOGNITION, MANUAL_STATE_RECOGNITION, MODEL_GEN, HELP,

        SAMPLE,
        WEB, EVALUATION, TESTING, PROCESS_DATA
    }

    @Parameter(key = "runtype", description = "Type of run (default instrument)", hasArgs = true, category = "Common")
    public static RunType RUN_TYPE = RunType.INSTRUMENT;

    @Parameter(key = "processPlayback", description = "Should frames be " +
            "processed during playback?",
            hasArgs = false, category = "Leap Motion Sampling")
    public static boolean PROCESS_PLAYBACK = false;

    @Parameter(key = "processScreenshots", description = "Should screenshots be processed during playback?",
            hasArgs = false, category = "Leap Motion Sampling")
    public static boolean PROCESS_SCREENSHOTS = false;

    @Parameter(key = "singleDataPool", description = "Should a single data pool be used to reconstruct hands?",
            hasArgs = false, category = "Leap Motion Sampling")
    public static boolean SINGLE_DATA_POOL = false;

    @Parameter(key = "skipDependencyTree", description = "Skip building dependency tree", hasArgs = false, category = "Leap Motion Testing")
    public static boolean SKIP_DEPENDENCY_TREE = false;

    @Parameter(key = "dependencyTreeOverride", description = "Use to always build a fresh dependency tree", hasArgs = false, category = "Leap Motion Testing")
    public static boolean DEPENDENCY_TREE_OVERRIDE = false;


    public void setOptions(CommandLine cmd) throws IllegalAccessException {
        try {
            for (String s : annotationMap.keySet()) {
                Parameter p = annotationMap.get(s);
                if (p.hasArgs()) {
                    String value = cmd.getOptionValue(p.key());
                    if (value != null) {
                        if (value.startsWith("prompt")){
                            String msg = "Please enter " + p.key() + " value";
                            if (value.contains(":")){
                                msg = value.split(":")[1];
                            }
                            value = JOptionPane.showInputDialog(msg);
                        }
                        setParameter(p.key(), value);
                    }
                } else {
                    if (cmd.hasOption(p.key())) {
                        setParameter(p.key(), Boolean.toString(true));
                    }
                }
            }

            Gson g = new Gson();
            File branches = new File(Properties.TESTING_OUTPUT + "/branches.csv");
            if (branches.getAbsoluteFile().exists()) {
                String branchesString = FileHandler.readFile(branches);
                Type mapType = new TypeToken<Map<Integer, Map<Integer, BranchHit>>>() {
                }.getType();

                try {
                ClassAnalyzer.setBranches((Map<Integer, Map<Integer, BranchHit>>) g.fromJson(branchesString, mapType));
                } catch (JsonSyntaxException e){}

                // App.out.println("- Found branches file at: " + branches.getAbsolutePath());
            }

            File linesFile = new File(Properties.TESTING_OUTPUT + "/lines.csv");
            if (linesFile.getAbsoluteFile().exists()) {
                try {
                    String linesString = FileHandler.readFile(linesFile);

                    Type mapType = new TypeToken<Map<Integer, Map<Integer, LineHit>>>() {
                    }.getType();

                    try {
                        ClassAnalyzer.setLines((Map<Integer, Map<Integer, LineHit>>) g.fromJson(linesString, mapType));
                    } catch (JsonSyntaxException e){}

                    //App.out.println("- Found lines file at: " + linesFile.getAbsolutePath());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            File relatedFile = new File(Properties.TESTING_OUTPUT + "/related_classes.csv");
            if (relatedFile.getAbsoluteFile().exists()) {
                String[] classes = FileHandler.readFile(relatedFile).split("\n");
                ArrayList<ClassTracker> clas = new ArrayList<ClassTracker>(classes.length - 1);
                for (int i = 1; i < classes.length; i++) {
                    if (classes[i].length() > 0) {
                        String[] clInfo = classes[i].split(",");
                        int lines = Integer.parseInt(clInfo[1]);
                        App.relatedLines += lines;
                        int brans = Integer.parseInt(clInfo[2]);
                        App.relatedBranches += (brans * 2);
                        clas.add(new ClassTracker(clInfo[0], lines, brans));
                    }
                }
                App.relatedClasses = clas;

                //App.out.println("- Found related classes file at: " + linesFile.getAbsolutePath());
                //App.out.println("[" + App.relatedLines + " related lines, " + App.relatedBranches + " related branches]");
            }
            if (Properties.INPUT_STRING != null) {
                Properties.INPUT = Properties.INPUT_STRING.split(";");
            }
            if (BEZIER_POINTS <= 1) {
                SWITCH_TIME = 1;
                BEZIER_POINTS = 2;
            }

            if (TUNING_PARAMETER != null) {
                Parameter p = annotationMap.get(TUNING_PARAMETER);
                String value = "" + (MIN_TUNING_VALUE + (Math.random() * (MAX_TUNING_VALUE - MIN_TUNING_VALUE)));
                App.out.println("- Tuning: " + p.key() + "=" + value);
                setParameter(p.key(), value);
            }

            OUTPUT_EXCLUDES_ARRAY = new ArrayList<String>();
            if (OUTPUT_EXCLUDES != null && OUTPUT_EXCLUDES.length() > 0) {
                OUTPUT_EXCLUDES_ARRAY.addAll(Arrays.asList(OUTPUT_EXCLUDES.split(",")));
            }

            OUTPUT_INCLUDES_ARRAY = new ArrayList<String>();
            if (OUTPUT_INCLUDES != null && OUTPUT_INCLUDES.length() > 0) {
                OUTPUT_INCLUDES_ARRAY.addAll(Arrays.asList(OUTPUT_INCLUDES.split(",")));
            }

            if (FORBIDDEN_PACKAGES_STRING != null) {
                FORBIDDEN_PACKAGES = FORBIDDEN_PACKAGES_STRING.split(";");
            }

            if (EXCLUDED_PACKAGES_STRING != null) {
                ArrayList<String> forbidden = new ArrayList<String>();
                String[] excluded = EXCLUDED_PACKAGES_STRING.split(";");

                for (String s : excluded) {
                    forbidden.add(s);
                }

                for (String s : FORBIDDEN_PACKAGES) {
                    forbidden.add(s);
                }

                FORBIDDEN_PACKAGES = new String[forbidden.size()];

                forbidden.toArray(FORBIDDEN_PACKAGES);
            }


            File lastRunDump = new File(Properties.TESTING_OUTPUT + "/current_run" + Properties.CURRENT_RUN + ".nmDump");

            if (lastRunDump.exists()) {

                String linesString = null;
                try {
                    linesString = FileHandler.readFile(lastRunDump);
                } catch (IOException e) {
                    e.printStackTrace(App.out);
                }

                String[] lines = linesString.split("\n");

                boolean acceptRestoration = false;

                for (String l : lines) {
                    l = l.trim();

                    if (l.startsWith("#")) {
                        continue;
                    }

                    String opt = l.substring(0, l.indexOf(":"));

                    String val = l.substring(l.indexOf(":") + 1);
                    if (opt.equalsIgnoreCase("frameGenerator"))
                        try {
                            acceptRestoration = FRAME_SELECTION_STRATEGY.equals(FrameSelectionStrategy.valueOf(val));
                        } catch (Exception e) {
                            acceptRestoration = false;
                        }
                }

                if (acceptRestoration) {

                    for (String l : lines) {

                        l = l.trim();

                        if (l.startsWith("#")) {
                            continue;
                        }

                        String opt = l.substring(0, l.indexOf(":"));

                        String val = l.substring(l.indexOf(":") + 1);


                        switch (opt) {
                            case "lines": {
                                Type mapType = new TypeToken<Map<Integer, Map<Integer, LineHit>>>() {
                                }.getType();

                                ClassAnalyzer.setLines((Map<Integer, Map<Integer, LineHit>>) g.fromJson(val, mapType));
                            }
                            break;

                            case "branches": {
                                Type mapType = new TypeToken<Map<Integer, Map<Integer, BranchHit>>>() {
                                }.getType();

                                ClassAnalyzer.setBranches((Map<Integer, Map<Integer, BranchHit>>) g.fromJson(val, mapType));
                            }
                            break;

                            case "current-run": {
                                Properties.CURRENT_RUN = Long.parseLong(val);
                            }

                            break;

                            case "remaining-budget": {
                                Properties.REMAINING_BUDGET = Long.parseLong(val);
                            }

                            break;

                            default: {
                                //do nothing
                            }

                            break;

                        }

                    }

                    App.out.println("--- Resuming after system halt (" + REMAINING_BUDGET + " budget)");

                    App.getApp().start();
                }
            }

            if (RESUME_RUN >= 0) {
                CURRENT_RUN = RESUME_RUN;
            }

            if (CURRENT_RUN == -1){
                CURRENT_RUN = System.currentTimeMillis();
            }
        } catch (Exception e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace(App.out);
        }
    }

    public Csv toCsv() {
        Csv csv = new Csv();
        for (String s : annotationMap.keySet()) {
            if (OUTPUT_EXCLUDES_ARRAY != null && OUTPUT_EXCLUDES_ARRAY.size() >
                    0 &&
                    OUTPUT_EXCLUDES_ARRAY.contains(s)) continue;

            if (OUTPUT_INCLUDES_ARRAY != null && OUTPUT_INCLUDES_ARRAY.size() >
                    0 &&
                    !OUTPUT_INCLUDES_ARRAY.contains(s)) continue;


            Field f = parameterMap.get(s);
            Class<?> cl = f.getType();

            String value = "";
            try {

                if (cl.isAssignableFrom(Number.class) || cl.isPrimitive()) {
                    if (cl.equals(Long.class) || cl.equals(long.class)) {
                        value = "" + f.getLong(null);
                    } else if (cl.equals(Double.class) || cl.equals(double.class)) {
                        value = "" + f.getDouble(null);
                    } else if (cl.equals(Float.class) || cl.equals(float.class)) {
                        value = "" + f.getFloat(null);
                    } else if (cl.equals(Integer.class) || cl.equals(int.class)) {
                        value = "" + f.getInt(null);
                    } else if (cl.equals(Boolean.class) || cl.equals(boolean.class)) {
                        value = "" + f.getBoolean(null);
                    }

                } else if (cl.isAssignableFrom(String.class) || f.getType().isEnum()) {
                    Object o = f.get(null);
                    if (o != null) {
                        value = o.toString();
                    } else {
                        value = NULL_VALUE_OUTPUT;
                    }
                }
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
            csv.add(s, value);
        }


        Field frameSelectionStrat = null;
        Field bezPoint = null;
        try {
            frameSelectionStrat = getClass().getField("FRAME_SELECTION_STRATEGY");
            bezPoint = getClass().getField("BEZIER_POINTS");
        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        }

        if (frameSelectionStrat == null || bezPoint == null) {
            throw new IllegalStateException("Cannot retrieve Frame Selection field.");
        }

        if (Properties.PLAYBACK_FILE != null) {
            csv.add(((Parameter) frameSelectionStrat.getAnnotations()[0]).key(), "USER_PLAYBACK");
        }

        if (Properties.RECORDING) {
            csv.add(((Parameter) frameSelectionStrat.getAnnotations()[0]).key(), "MANUAL_TESTING");
        }


        if (Properties.SWITCH_TIME <= 1) {
            csv.add(((Parameter) bezPoint.getAnnotations()[0]).key(), "0");
        }

        csv.finalize();

        return csv;
    }

    public void setOptions(String[] args) {
        try {

//            if (!DIRECTORY.endsWith("/processed")) {
//                Properties.DIRECTORY += "/processed";
//            }

            Options options = new Options();

            for (String s : annotationMap.keySet()) {
                Parameter p = annotationMap.get(s);
                options.addOption(p.key(), p.hasArgs(), p.description());
            }

            CommandLineParser parser = new BasicParser();
            CommandLine cmd = null;
            try {
                cmd = parser.parse(options, args);
            } catch (UnrecognizedOptionException e) {

                App.out.println(e.getLocalizedMessage());
                System.exit(-1);
            }

            setOptions(cmd);


        } catch (Exception e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace(App.out);
        }
    }

    public void printOptions() {
        for (String s : categoryMap.keySet()) {
            App.out.println(s);
            for (String opt : categoryMap.get(s)) {
                Parameter p = annotationMap.get(opt);
                String opts = " ";
                if (p.hasArgs()) {
                    opts = ":[arg] ";
                }
                App.out.println(" -" + p.key() + opts + " #" + p.description() + ".");
            }
        }
    }


    public void printOptionsMd() {

        App.out.println("# Runtime Options");
        App.out.println("| Key | Description |");
        App.out.println("| --- | --- |");
        for (String s : categoryMap.keySet()) {
            App.out.println("| **" + s + "** |  |");

            for (String opt : categoryMap.get(s)) {
                Parameter p = annotationMap.get(opt);
                String opts = " ";
                if (p.hasArgs()) {
                    opts = ":[arg] ";
                }
                App.out.println("| " + p.key() + opts + " | _" + p.description()
                        + "_ |");
            }
        }
    }

    private static Properties instance;

    public static Properties instance() {
        if (instance == null) {
            instance = new Properties();
        }
        return instance;
    }


}
