package com.sheffield.leapmotion.runtypes.interaction;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.output.StateComparator;
import com.sheffield.leapmotion.util.FileHandler;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URI;
import java.util.*;
import java.util.List;

/**
 * Created by thoma on 21/06/2017.
 */
public class DeepQNetworkInteraction extends UserInteraction {
    private long minTime = Long.MAX_VALUE;


    private static final float RANDOM_PROBABILITY = 0.05f;
    public static final int ITERATIONS_BEFORE_RESET = 10;
    public static final int SAMPLE = 30;
    public static final int MIN_SMAPLE = 10;
    public static final float DIMINISH = 0.2f;
    public static final int epoch = 30;


    private static State lastState;

    private HashMap<Integer, HashMap<Integer, EventReward>> eventRewards;
    private HashMap<Integer, HashMap<Integer, EventReward>> futureEventRewards;
    private ArrayList<EventState> dictionary;
    private HashMap<Integer, State> eventStates;
    private File dictionaryFile;

    private HashMap<Integer, State> states;

    private int steps = 0;

    private MultiLayerNetwork net;

    private int iteration = 0;
    private File statesFile;

    Gson gson;

    @Override
    public void load() throws IOException {
        super.load();

        lastState = State.ORIGIN;

        eventStates = new HashMap<>();

        states = new HashMap<>();

        eventRewards = new HashMap<>();
        futureEventRewards = new HashMap<>();


        dictionaryFile = new File(Properties.TESTING_OUTPUT + "/dict.json");

        statesFile = new File(Properties.TESTING_OUTPUT + "/states.json");


        gson = new Gson();

        if (!statesFile.exists()){
            if (!statesFile.getParentFile().exists()){
                statesFile.getParentFile().mkdirs();
            }
            statesFile.createNewFile();
        }

        lastState = captureState(Event.NONE);

        dictionary = new ArrayList<>();
        if (dictionaryFile.exists() && !Properties.IGNORE_DICTIONARY && statesFile.exists()) {
            String[] lns = FileHandler.readFile(dictionaryFile).split("\n");
            for (String line : lns) {
                    if (line.length() > 0) {
                    dictionary.add(gson.fromJson(line, EventState.class));
                }
            }

            String[] statesString = FileHandler.readFile(statesFile).split("\n");

            eventStates = new HashMap<>();

            for (String s : statesString) {
                State state = gson.fromJson(s, State.class);
                states.put(state.getStateNumber(), state);
                eventStates.put(state.getStateNumber(), state);
            }
        } else if (!dictionaryFile.exists()) {
            if (!dictionaryFile.getParentFile().exists()) {
                dictionaryFile.getParentFile().mkdirs();
            }
            dictionaryFile.createNewFile();
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(System.currentTimeMillis())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(1e-3)
                .regularization(true)
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .activation(Activation.IDENTITY)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(StateComparator.X_SIZE * StateComparator.Y_SIZE).nOut(StateComparator.X_SIZE)
                        .build())
//                .layer(1, new DenseLayer.Builder().nIn(StateComparator.X_SIZE * StateComparator.Y_SIZE).nOut(StateComparator.X_SIZE)
//                        .build())
                .layer(1, new DenseLayer.Builder().nIn(StateComparator.X_SIZE).nOut(StateComparator.Y_SIZE)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(StateComparator.Y_SIZE).nOut(rawEvents.size()).build())
                .backprop(true)
                .pretrain(false)
                .build();

        net = new MultiLayerNetwork(conf);

        net.init();
//
//        UIServer uiServer = UIServer.getInstance();

        StatsStorage ss = new InMemoryStatsStorage();

        net.setListeners(new StatsListener(ss), new ScoreIterationListener(100));
//
//        uiServer.attach(ss);

        if (Properties.SHOW_DISPLAY) {
            Desktop.getDesktop().browse(URI.create("http://localhost:9000"));
        }

        lastState = captureState(Event.NONE);

        //TODO: Load any database info from previous runs
    }

    private boolean debugHeader = true;

    @Override
    public Event interact(long timePassed) {

        Event e;

        if (rawEvents.size() == 0) {
            return Event.NONE;
        }

        if (Math.random() <= RANDOM_PROBABILITY) {
            e = rawEvents.get((int) (Math.random() * rawEvents.size()));
        } else {
            double[] img = lastState.getImage();
            INDArray preds = net.output(Nd4j.create(img));

            float max = 0f;
            int index = 0;

            String cont = "";

            for (int i = 0; i < preds.length(); i++) {
                float m = preds.getFloat(i);
                if (m > max) {
                    max = m;
                    index = i;
                }
                cont += i + "," + iteration + "," + m + "," + Properties.IGNORE_DICTIONARY + "," + dictionary.size() + "," + lastState.getStateNumber() + "," + states.size() + "\n";
            }

            File f = new File(Properties.TESTING_OUTPUT + "/debug.csv");

            if (f.exists()) {

                if (debugHeader) {
                    try {
                        if (FileHandler.readFile(f).length() == 0) {
                            FileHandler.writeToFile(f, "output,iteration,value,ignore_dict,dict_size,state,states_size\n");
                        }
                    } catch (IOException e1) {
                        e1.printStackTrace();
                    }
                    debugHeader = false;
                }

                try {
                    FileHandler.appendToFile(f, cont);
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }

            App.out.println(index + ": " + max);

            e = rawEvents.get(index);
        }

        return e;
    }

    @Override
    public void postInteraction(Event e) {
        super.postInteraction(e);

        State state = captureState(e);

        steps = (steps++) % ITERATIONS_BEFORE_RESET;

        float observedReward; // = states.size() == state.getStateNumber() ? 1 : 0

        if (!states.containsKey(lastState.getStateNumber())) {
            states.put(lastState.getStateNumber(), lastState);

            StateComparator.captureState(lastState.getImage(), lastState.getStateNumber());

            try {
                FileHandler.appendToFile(statesFile, gson.toJson(lastState) + "\n");
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }

        if (lastState.getStateNumber() == -1 || eventStates.get(lastState.getStateNumber()).getImage() == null) {
            lastState = state;
            return;
        }

        if (!eventRewards.containsKey(lastState.getStateNumber())) {
            eventRewards.put(lastState.getStateNumber(), new HashMap<>());
        }

        EventReward erew = null;


        if (!eventRewards.get(lastState.getStateNumber()).containsKey(e.getEventIndex())) {
            observedReward = 1;
            eventRewards.get(lastState.getStateNumber()).put(e.getEventIndex(), new EventReward(e, 0));

            EventState es = new EventState(lastState, e, state.getImage(), observedReward);

            if (!dictContains(es)) {

                dictionary.add(es);
                try {
                    if (!Properties.IGNORE_DICTIONARY) {
                        FileHandler.appendToFile(dictionaryFile, gson.toJson(es) + "\n");
                    }
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }

            if (!states.containsKey(state.getStateNumber())) {
                states.put(state.getStateNumber(), state);
                StateComparator.captureState(state.getImage(), state.getStateNumber());
                try {
                    FileHandler.appendToFile(statesFile, gson.toJson(state) + "\n");
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }

        } else {
            observedReward =  lastState.getStateNumber() == state.getStateNumber() ? 1 : eventRewards.get(lastState.getStateNumber()).get(e.getEventIndex()).getReward()-1;
        }


        eventRewards.get(lastState.getStateNumber()).get(e.getEventIndex()).setReward(observedReward);


        int sample = Math.min(dictionary.size(), MIN_SMAPLE + (int) (Math.random() * (SAMPLE - MIN_SMAPLE)));

        float[][] futureReward = new float[sample][rawEvents.size()];

        double[][] inpts = new double[sample][StateComparator.Y_SIZE * StateComparator.X_SIZE];


        for (int i = 0; i < sample; i++) {
            int index = (int) Math.round(Math.random() * (dictionary.size() - 1));

            EventState es = dictionary.get(index);

            if (es.getStateNumber() == -1 || eventStates.get(es.getLastState()).getImage() == null) {
                i--;
                continue;
            }

            inpts[i] = eventStates.get(es.getLastState()).getImage();

            assert inpts[i].length == StateComparator.X_SIZE * StateComparator.Y_SIZE;


            if (es.getNewState() != null && es.getNewState().length > 0 && i < dictionary.size() - 1) {

                if (!futureEventRewards.containsKey(es.getLastState())) {
                    futureEventRewards.put(es.getLastState(), new HashMap<>());
                    for (Event re : rawEvents) {
                        futureEventRewards.get(es.getLastState()).put(re.getEventIndex(), new EventReward(re, 1f));
                    }
                }

                if (eventRewards.containsKey(es.getLastState())) {
                    for (Integer i2 : eventRewards.get(es.getLastState()).keySet()) {
                        futureEventRewards.get(es.getLastState()).put(i, eventRewards.get(es.getLastState()).get(i2));
                    }
                }

                float max = 0f;

                for (EventReward er : futureEventRewards.get(es.getLastState()).values()) {
                    if (er.getReward() > max) {
                        max = er.getReward();
                    }
                }

                for (int j = 0; j < futureEventRewards.get(es.getLastState()).size(); j++) {
                    futureReward[i][j] =
                            futureEventRewards.get(es.getLastState()).get(j).getReward() + (DIMINISH * max);
                }

            }

        }

        if (sample > 0) {
            try {
                INDArray inputs = Nd4j.create(inpts);
                INDArray outputs = Nd4j.create(futureReward);

                DataSet allData = new DataSet(inputs, outputs);

                ListDataSetIterator ldsi = new ListDataSetIterator(allData.asList(), sample);

                for (int i = 0; i < epoch; i++){
                    net.fit(ldsi);
                }
            } catch (NullPointerException exc) {
                exc.printStackTrace(App.out);
            }
        }

        if (steps == 0) {
//            for (State s : futureEventRewards.keySet()){
//                eventRewards.put(s, futureEventRewards.get(s));
//            }
            futureEventRewards.clear();
        }

        lastState = state;

        iteration++;

    }

    public State captureState(Event e) {
        double[] newImage = StateComparator.screenshotState();

        int stateNumber = states.size();

        boolean found = false;

        State state = State.ORIGIN;

        for (State s : eventStates.values()) {
            if (s.screenshotIdentical(newImage)) {
                state = s;
                found = true;
                break;
            }
        }

        if (!found) {
            state = new State(stateNumber, newImage, lastState.getStateNumber());
            states.put(states.size(), state);
            StateComparator.captureState(state.getImage(), state.getStateNumber());

            try {
                FileHandler.appendToFile(statesFile, gson.toJson(state) + "\n");
            } catch (IOException e1) {
                e1.printStackTrace();
            }
            eventStates.put(stateNumber, state);
        }

        return state;
    }

    public boolean dictContains(EventState es) {

        ArrayList<Integer> seenStatesNotEqual = new ArrayList<>();
        for (EventState e : dictionary) {
            if (e.getLastEvent().equals(es.getLastEvent())) {
                if (seenStatesNotEqual.contains(e.getStateNumber())) {
                    continue;
                } else {
                    if (e.screenshotIdentical(es.getImage())) {
                        return true;
                    } else {
                        seenStatesNotEqual.add(e.getStateNumber());
                    }
                }
            }
        }
        return false;
    }
}
