package com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram;

import com.sheffield.leapmotion.util.FileHandler;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Thomas on 06-02-2017.
 */
public class NGramModel implements Serializable {


    public static final String DELIMITER = " ";

    public static NGram getNGram(int n, File f) throws IOException {
        return getNGram(n, FileHandler.readFile(f));
    }

    public static NGram getNGram(){
        return new NGram("");
    }

    public static NGram getNGram(int n, String text){

        NGram root = new NGram("");

        String[] words = text.split(DELIMITER);

        for (int i = 0; i < words.length - n + 1; i++){
            String t = "";

            for (int j = 0; j < n; j++){
                String word = words[i+j];
                t += DELIMITER + word;
            }
            t = t.substring(t.indexOf(DELIMITER)+1);
            root.add(t);
        }

        return root;

    }

    public static NGram getNGramOpenVocab(int n, String text) {

        NGram root = new NGram("");

        String[] words = text.split(DELIMITER);

        List<String> seen = new ArrayList<String>();

        for (int i = 0; i < words.length - n + 1; i++) {
            String t = "";

            for (int j = 0; j < n; j++) {
                String word = words[i + j];
                if (!seen.contains(word)) {
                    seen.add(word);
                    word = NGram.UNKNOWN;
                }
                t += DELIMITER + word;
            }
            t = t.substring(t.indexOf(DELIMITER) + 1);
            root.add(t);
        }
        return root;
    }

}
