package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.util.FileHandler;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.runtypes.state_identification.ImageStateIdentifier;
import com.sheffield.leapmotion.runtypes.state_identification.StateShowingFrame;
import com.sheffield.output.Csv;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by thomas on 18/11/2016.
 */
public class StateRecognisingRunType implements RunType {

    ImageStateIdentifier isi;

    public StateRecognisingRunType(ImageStateIdentifier isi){
        this.isi = isi;
    }

    @Override
    public int run() {
        processStates(isi);
        return 0;
    }

    public static void processStates(ImageStateIdentifier isi) {
        try {
            //INPUT should be a directory contaning screenshots
            String directory = Properties.INPUT[0];

            File dir = new File(directory);

            int counter = 1;

            Properties.FRAME_SELECTION_STRATEGY = Properties
                    .FRAME_SELECTION_STRATEGY.NONE;

            Properties.CURRENT_RUN = 0;

            final HashMap<Integer, BufferedImage> seenStates = new HashMap<Integer, BufferedImage>();

            final int STATES_PER_ROW = 5;

            JFrame imageStates = new JFrame("Seen States") {
                @Override
                public void paint(Graphics g) {
                    //super.paint(g);

                    Graphics2D g2d = (Graphics2D) g;

                    g2d.clearRect(0, 0, getWidth(), getHeight());

                    Set<Integer> keys = seenStates.keySet();

                    Integer[] ks = new Integer[keys.size()];

                    keys.toArray(ks);

                    Arrays.sort(ks);

                    int width = getWidth() / STATES_PER_ROW;

                    int height = (width * 9) / 16;

                    for (int i = 0; i < ks.length; i++) {
                        int x = (i % STATES_PER_ROW) * width;

                        int y = (i / STATES_PER_ROW) * height;

                        if (ks[i] == null) {
                            g2d.setColor(Color.RED);
                            g2d.fillRect(x, y, width, height);
                            continue;
                        }

                        int key = ks[i];

                        g2d.drawImage(seenStates.get(key), x, y, width, height, null);

                        g2d.setColor(Color.red);

                        g2d.drawString("" + key, x + 10, y + height / 2);


                    }

                }
            };

            imageStates.setSize(800, 600);

            imageStates.setLocation(960, 0);

            StateShowingFrame image = new StateShowingFrame();
            image.setSize(960, 540);

            image.setLocation(0, 0);

            image.setVisible(true);

            imageStates.setVisible(true);


            File[] files = dir.listFiles();

            String[] fs = new String[files.length];

            for (int i = 0; i < files.length; i++) {
                fs[i] = files[i].getAbsolutePath();
            }

            //Arrays.sort(fs);

            Comparator<String> comp = new Comparator<String>() {
                @Override
                public int compare(String o1, String o2) {
                    int n = o1.lastIndexOf("SCREEN") + 6;
                    int n1 = o2.lastIndexOf("SCREEN") + 6;

                    if (n == n1 && n == 5) {
                        return 0;
                    }

                    if (n == 5) {
                        return 1;
                    }

                    if (n1 == 5) {
                        return -1;
                    }

                    int s = Integer.parseInt(o1.substring(n, o1.length() - 4));
                    int s1 = Integer.parseInt(o2.substring(n, o2.length() - 4));

                    return s - s1;
                }
            };

            Arrays.sort(fs, comp);

            for (String s : fs) {

                File f = new File(s);

                if (f.isDirectory()) {
                    continue;
                }

                if (!f.getName().toLowerCase().endsWith(".png")) {
                    continue;
                }

                BufferedImage orig = ImageIO.read(f);

                BufferedImage bi = new BufferedImage(orig.getWidth(), orig.getHeight(), BufferedImage.TYPE_INT_RGB);

                Graphics2D g2 = bi.createGraphics();

                g2.drawImage(orig, 0, 0, null);

                g2.dispose();

                image.setImage(bi);

                image.paint(image.getGraphics());

                int state = isi.identifyImage(bi, seenStates);

                if (!seenStates.containsKey(state)) {
                    seenStates.put(state, bi);

                    imageStates.paint(imageStates.getGraphics());
                }

                Csv csv = new Csv();

                csv.add("imageId", f.getName());

                csv.add("stateAssignment", "" + state);

                csv.add("totalStates", "" + seenStates.keySet().size());

                csv.add("counter", "" + counter);

                csv.add("histogramBins", "" + Properties.HISTOGRAM_BINS);

                csv.add("histogramThreshold", "" + Properties.HISTOGRAM_THRESHOLD);



                csv.finalize();

                File csvFile = new File(f.getParentFile().getAbsolutePath(), Properties.TESTING_OUTPUT + "/states/" + isi.getOutputFilename());
                if (csvFile.getParentFile() != null) {
                    csvFile.getParentFile().mkdirs();
                }
                try {
                    boolean newFile = !csvFile.getAbsoluteFile().exists();
                    String contents = "";

                    if (newFile) {
                        csvFile.createNewFile();
                        contents += csv.getHeaders() + "\n";
                    }

                    contents += csv.getValues() + "\n";

                    FileHandler.appendToFile(csvFile, contents);

                } catch (IOException e) {
                    e.printStackTrace();
                }
                counter++;
            }

            imageStates.setVisible(false);

            imageStates = null;

            image.setVisible(false);

            image = null;
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

}
