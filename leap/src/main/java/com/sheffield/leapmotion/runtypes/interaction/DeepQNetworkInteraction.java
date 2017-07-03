package com.sheffield.leapmotion.runtypes.interaction;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.output.StateComparator;
import com.sheffield.leapmotion.sampler.MouseEvent;
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
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
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

    private Event lastEvent = Event.NONE;

    private int iteration = 0;

    Gson gson;

    @Override
    public void load() throws IOException {
        super.load();

        lastState = State.ORIGIN;
        states = new HashMap<>();

        gson = new Gson();

        lastState = captureState(Event.NONE);

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

            String input = "";

            for (double d : img)
                input += d + " ";

            input += lastEvent.toCsv().replace(",", " ");

            try {
                String pythonCommand = "python tensor_play.py " + input;
                Process process = Runtime.getRuntime().exec(pythonCommand);
                BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
                BufferedReader be = new BufferedReader(new InputStreamReader(process.getErrorStream()));

                String line;

                e = Event.NONE;

                while ((line = br.readLine()) != null && line.trim().length() > 0) {
                    //output: [[ x y lmm rmm]]
                    line = line.replace("[[", "");
                    line = line.replace("]]", "");
                    //x y lmm rmm
                    String[] eles = line.split(" ");
                    float lmm = Float.parseFloat(eles[2]);
                    float rmm = Float.parseFloat(eles[3]);

                    MouseEvent me = MouseEvent.MOVE;


                    if (lmm > rmm && lmm > 0.5) {
                        me = MouseEvent.LEFT_DOWN;
                    } else if (rmm > lmm && rmm > 0.5) {
                        me = MouseEvent.RIGHT_DOWN;
                    } else if (lmm < rmm && lmm < -0.5) {
                        me = MouseEvent.LEFT_UP;
                    } else if (rmm < lmm && rmm < -0.5) {
                        me = MouseEvent.RIGHT_UP;
                    }

                    e = new Event(me,
                            (int) (Float.parseFloat(eles[0]) * Event.bounds.getWidth()),
                            (int) (Float.parseFloat(eles[1]) * Event.bounds.getHeight()),
                            System.currentTimeMillis(),
                            iteration);
                }

                while ((line = be.readLine()) != null){
                    App.out.println(line);
                }

            } catch (IOException e1) {
                e1.printStackTrace();

                e = Event.NONE;
            }

        }

        return e;
    }

    @Override
    public void postInteraction(Event e) {
        super.postInteraction(e);

        State state = captureState(e);

        lastState = state;

        iteration++;

    }

    public State captureState(Event e) {
        double[] newImage = StateComparator.screenshotState();

        int stateNumber = states.size();

        State state = new State(stateNumber, newImage, lastState.getStateNumber());
        StateComparator.captureState(state.getImage(), state.getStateNumber());

        return state;
    }
}
