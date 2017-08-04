package com.thomasdeanwhite.gui.ngram;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Thomas on 06-02-2017.
 */
public class NGram implements Serializable {

    public static final String UNKNOWN = "<unk>";
    protected String element;

    protected int count = 0;
    protected float probability = 0f;

    protected ArrayList<NGram> children;

    protected int n = 0;

    protected boolean finalised = false;
    public NGram (String label){
        children = new ArrayList<NGram>();
        element = label;
    }


    public void add(String text){
        block();
        NGram child = null;

        if (!text.contains(NGramModel.DELIMITER)){
            child = restore(text);
        } else {
            String[] newText = text.split(NGramModel.DELIMITER);
            child = restore(newText[0]);

            if (n < newText.length){
                n = newText.length;
            }

            String childText = text.substring(text.indexOf(NGramModel.DELIMITER)+1);

            if (childText.length() > 0) {
                child.add(childText);
            }
        }

        child.increment();

        if (!children.contains(child)) {
            children.add(child);
        }
    }

    private NGram restore(String s){
        for (NGram n : children){
            if (n.element.equals(s)){
                return n;
            }
        }
        return new NGram(s);
    }


    public void increment(){
        block();
        count++;
    }

    private void block(){
//        if (finalised){
//            throw new IllegalStateException("Cannot modify NGram once finalised.");
//        }
    }

    public void calculateProbabilities(){

        if (finalised){
            return;
        }

        calculateProbabilitiesChildren();

        finalised = true;
    }

    private void calculateProbabilitiesChildren(){
        int total = 0;

        for (NGram n: children){
            n.calculateProbabilitiesChildren();
            total += n.count;
        }

        for (NGram n: children){
            n.probability = n.count / (float) total;
        }
    }

    private String toLine(String current){
        current += NGramModel.DELIMITER + element;
        if (children.size() == 0){
            return current;
        }
        String s = "";
        for (NGram n : children){
            s += n.toLine(current).trim() + "\n";
        }
        return s;
    }

    public float getProbability(String child){

        String[] cs = child.split(NGramModel.DELIMITER);

        NGram unk = null;

        for (NGram n : children){
            if (n.element.equals(cs[0])){
                return n.getProbability(child.substring(child.indexOf(NGramModel.DELIMITER)+1));
            }

            if (n.element.equals(UNKNOWN)){
                unk = n;
            }
        }



        return unk == null ? probability : unk.getProbability(child.substring(child.indexOf(NGramModel.DELIMITER)+1));
    }

    public int getCount(String child){

        String[] cs = child.split(NGramModel.DELIMITER);

        for (NGram n : children){
            if (n.element.equals(cs[0])){
                return n.getCount(child.substring(child.indexOf(NGramModel.DELIMITER)+1));
            }
        }

        return count;
    }

    public float getProbability(){
        return probability;
    }

    public String babbleNext(String text){

        if (!finalised){
            calculateProbabilities();
        }

        String babble = babbleRecursive(text);

        String[] elements = null;
        String returnString = null;

        if (babble == null || babble.length() == 0){
            if (!text.contains(NGramModel.DELIMITER)){
                throw new DataSparsityException("Data is too sparse!");
            }
            String newText = text.substring(text.indexOf(NGramModel.DELIMITER)+1);
            babble = babbleRecursive(newText);
            if (babble == null || babble.length() == 0){
                returnString = null;
            } else {
                elements = babble.split(NGramModel.DELIMITER);
                for (int i = elements.length - 1; i >= 0 && i > elements.length - n; i--) {
                    returnString = NGramModel.DELIMITER + elements[i] + returnString;
                }
            }
        } else {

            elements = babble.split(NGramModel.DELIMITER);
            returnString = "";

            for (int i = elements.length - 1; i >= 0 && i > elements.length - n; i--) {
                returnString = NGramModel.DELIMITER + elements[i] + returnString;
            }

            while (returnString.charAt(0) == NGramModel.DELIMITER.charAt(0)) {
                returnString = returnString.substring(1);
            }
        }

        if (returnString == null){
            throw new DataSparsityException("Data is too sparse!");
        }

        return returnString;
    }

    private String babbleRecursive(String text){
        if (text.length() == 0){
            float probability = (float) Math.random();

            for (NGram c : children){
                probability -= c.getProbability();

                if (probability <= 0.0000001){
                    return NGramModel.DELIMITER + c.element;
                }
            }
            return null;
        }
        String childText = text;
        String recurringText = "";
        if (text.contains(NGramModel.DELIMITER)) {
            String[] newText = text.split(NGramModel.DELIMITER);
            childText = newText[0];
            if (text.startsWith(NGramModel.DELIMITER)){
                childText = newText[1];
            }

            recurringText = text.substring(text.indexOf(NGramModel.DELIMITER)+1);
        }
        NGram child = restore(childText);

        String childBabble = child.babbleRecursive(recurringText);

        if (childBabble == null){
            return null;
        }

        return NGramModel.DELIMITER + child.element + childBabble;
    }

    public String toString(){

        String ngram = "";

        for (NGram n : children){
            ngram += n.toLine(element);
        }

        return ngram;
    }


    public void merge(NGram ng){
        //TODO: Implement NGram data Merge

        mergeChildren(ng);

        calculateProbabilitiesChildren();
    }

    public void mergeChildren(NGram ng){
        //TODO: Implement NGram data Merge

        if (ng.element.equals(element)){
            count += ng.count;

            if (ng.children.size() != 0){
                for (NGram on : ng.children){
                    boolean matched = false;
                    for (NGram n : children){
                        if (n.element.equals(on.element)) {
                            n.mergeChildren(on);
                            matched = true;
                        }
                    }

                    //add new child if doens't exist
                    if (!matched){
                        children.add(on);
                    }
                }
            }
        }
    }

}
