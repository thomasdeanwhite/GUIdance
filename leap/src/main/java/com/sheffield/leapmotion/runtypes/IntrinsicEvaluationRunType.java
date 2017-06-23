package com.sheffield.leapmotion.runtypes;

import com.google.gson.Gson;
import com.sheffield.leapmotion.*;

import static com.sheffield.leapmotion.Properties.*;

import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.clustering.ClusterResult;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.clustering.WekaClusterer;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram.NGram;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram.NGramModel;
import com.sheffield.leapmotion.output.StateComparator;
import com.sheffield.leapmotion.util.FileHandler;
import com.sheffield.leapmotion.util.ProgressBar;
import com.sheffield.output.Csv;
import weka.core.Instance;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by thomas on 08/02/17.
 */
public class IntrinsicEvaluationRunType implements RunType {
    private HashMap<String, ArrayList<Float>> perplexity = new HashMap<>();

    @Override
    public int run() {

        HashMap<String, String> files = new HashMap<String, String>();

        files.put("joint_positions_pool.ARFF", "joint_position");
        files.put("hand_positions_pool.ARFF", "hand_position");
        files.put("hand_rotations_pool.ARFF", "hand_rotation");
        files.put("hand_joints_pool.ARFF", "hand_joints");

        HashMap<String, ClusterResult> results = new HashMap<String, ClusterResult>(3);

        if (DIRECTORY.toLowerCase().endsWith("/processed")) {
            DIRECTORY = DIRECTORY.substring(0, DIRECTORY.lastIndexOf("/"));
        }
        String dataDir = DIRECTORY + "/" + INPUT[0];

        List<String> trainingFiles = new ArrayList<>();

        for (int i = 1; i < INPUT.length; i++){
            trainingFiles.add(DIRECTORY + "/" + INPUT[i]);
        }


        long seed = System.currentTimeMillis();

        files.keySet().stream().map(s -> {

            String joints = dataDir + "/" + s;

            List<String> fs = new ArrayList<>();

            for (String tf : trainingFiles){
                fs.add(tf + "/" + s);
            }

            WekaClusterer wc = new WekaClusterer(joints, fs);

            try {
                App.out.println(ProgressBar.getHeaderBar(21));
                App.out.print("\r" + ProgressBar.getProgressBar(21, 0f) +
                        " clustering...");

                ClusterResult cr = wc.cluster();

                results.put(s, cr);

                App.out.println();

                HashMap<String, String> assignments = cr.getAssignments();

                ArrayList<String> keys = new ArrayList<>();

                keys.addAll(assignments.keySet());

                keys.sort(new Comparator<String>() {
                    @Override
                    public int compare(String s, String t1) {
                        return s.compareTo(t1);
                    }
                });

                ArrayList<String> clusterOrder = new ArrayList<>();

                for (String key : keys){
                    clusterOrder.add(assignments.get(key));
                }

                NGram ng = NGramModel.getNGram();

                long trainingSize = 0l;

                for (String trainingFile : trainingFiles){


                    File outputSequence = new File(trainingFile + "/sequence_hand_data");

                    String outSeq = FileHandler.readFile(outputSequence);

                    String[] outpSeq = outSeq.split(",");

                    long dataSize = Long.parseLong(outpSeq[outpSeq.length-1].split("@")[0]) -
                            Long.parseLong(outpSeq[0].split("@")[0]);

                            outSeq = "";

                    trainingSize += dataSize;

                    for (String os : outpSeq){
                        outSeq += assignments.get(os + trainingFile + "/" + s) + " ";
                    }

                    NGram ngf = NGramModel.getNGramOpenVocab(N, outSeq);

                    ng.merge(ngf);
                }

                ng.calculateProbabilities();

                Csv csv = new Csv();

                Random r = new Random(seed);


                IntrinsicEvaluationRunType.this.perplexity.put(s, new ArrayList<>());


                File outputSequence = new File(dataDir+ "/sequence_hand_data");

                String[] outSeq = FileHandler.readFile(outputSequence).split(",");

                long dataSize = Long.parseLong(outSeq[outSeq.length-1].split("@")[0]) -
                        Long.parseLong(outSeq[0].split("@")[0]);

                float perplex = 0f;

                float lastProg = 0f;
                App.out.println(ProgressBar.getHeaderBar(21));
                for (int i = N; i <= outSeq.length; i++) {
                    String candidate = "";

                    for (int j = i - N; j < i; j++){
                        candidate += assignments.get(outSeq[j] + joints) + " ";
                    }
                    float probability = ng.getProbability(candidate);

//                    while (probability == 0f) {
//                        candidate = candidate.substring(candidate.indexOf(" ")+1);
//                        probability = ng.getProbability(candidate) * 0.0001f;
//                        if (candidate.length() == 0){
//                            probability = (float)Math.pow(0.0001, N);
//                        }
//                    }

//                    IntrinsicEvaluationRunType.this.perplexity.get(s).add((float)Math.pow(1f/probability,
//                            1/(float)N));
                    float perplexity = (float) Math.pow(1f / probability, 1 / (float) N);
                    perplex += perplexity;

                    float prog = i / (float) outSeq.length;

                    if (Properties.SHOW_PROGRESS || prog > lastProg){
                        App.out.print("\r" + ProgressBar.getProgressBar(21, prog) + candidate + ": " + perplexity);
                        lastProg = prog + 0.1f;
                    }
                }

                App.out.println();


                csv.add("preplexity", "" + perplex);
                csv.add("cluster", "" + CLUSTERS);
                csv.add("N", "" + N);
                csv.add("model", s);
                csv.add("dataPool", Properties.INPUT[0]);
                csv.add("trainingSize", ""+trainingSize);
                csv.add("dataSize", ""+dataSize);
                String tf = "";

                for (int i = 1; i < Properties.INPUT.length; i++){
                    tf += Properties.INPUT[i] + ";";
                }

                tf = tf.substring(0, tf.length()-1);
                csv.add("trainingData", tf);

                csv.finalize();

                File f = new File("NuiMimicEvaluation.csv");

                if (!f.exists()){
                    f.createNewFile();
                    FileHandler.writeToFile(f, csv.getHeaders() + "\n");
                }

                FileHandler.appendToFile(f, csv.getValues() + "\n");

                App.out.println(s + " perplexity: " + perplex);

            } catch (Exception e) {
                e.printStackTrace(App.out);
            }
            return s;
        }).collect(Collectors.toList());

        return 0;
    }
}
