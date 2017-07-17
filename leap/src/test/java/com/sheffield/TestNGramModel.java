package com.sheffield;

import com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram.DataSparsityException;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram.NGram;
import com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram.NGramModel;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Created by Thomas on 06-02-2017.
 */
public class TestNGramModel {

    private NGram ng;



    @Test
    public void createNGram(){
        String sentence = "That that is is that that is not is not Is that it It is".toLowerCase();

        ng = NGramModel.getNGram(3, sentence);

        assertEquals("that that is\n" +
                "that is is\n" +
                "that is not\n" +
                "that it it\n" +
                "is is that\n" +
                "is that that\n" +
                "is that it\n" +
                "is not is\n" +
                "not is not\n" +
                "not is that\n" +
                "it it is\n", ng.toString());
    }


    @Test
    public void checkProbabilities(){
        String sentence = "That that is is that that is not is not Is that it It is".toLowerCase();

        ng = NGramModel.getNGram(2, sentence);

        ng.calculateProbabilities();

        assertEquals(0.4, ng.getProbability("that that"), 0.000001f);
    }


    @Test
    public void checkMergeAddition(){
        String sentence = "That that is is that that is not is not Is that it It is".toLowerCase();

        ng = NGramModel.getNGram(2, sentence);

        ng.calculateProbabilities();

        assertEquals(2, ng.getCount("that that"));
        assertEquals(2, ng.getCount("that is"));
        assertEquals(1, ng.getCount("that it"));

        ng.merge(ng);

        assertEquals(4, ng.getCount("that that"));
        assertEquals(4, ng.getCount("that is"));
        assertEquals(2, ng.getCount("that it"));
    }

    @Test
    public void checkMergeAdditionOther(){
        String sentence = "George whilst Harry had had had had had had had had had had had a better effect on the teacher".toLowerCase();

        ng = NGramModel.getNGram(2, sentence);

        ng.calculateProbabilities();

        //ng.merge(ng);

        assertEquals(10, ng.getCount("had had"));
        assertEquals(0.90909094f, ng.getProbability("had had"), 0.000001f);

        sentence = "had Harry had what george had had this all time".toLowerCase();

        NGram ng2 = NGramModel.getNGram(2, sentence);

        ng2.calculateProbabilities();

        ng.merge(ng2);

        assertEquals(11, ng.getCount("had had"));
        assertEquals(0.25f, ng2.getProbability("had had"), 0.000001f);
        assertEquals(0.73333335f, ng.getProbability("had had"), 0.000001f);
    }

    @Test
    public void checkMergeProbability(){
        String sentence = "That that is is that that is not is not Is that it It is".toLowerCase();

        ng = NGramModel.getNGram(2, sentence);
        NGram ng2 = NGramModel.getNGram(2, sentence);

        ng.calculateProbabilities();
        ng2.calculateProbabilities();

        ng.merge(ng2);

        assertEquals(ng2.getProbability("that that"), ng.getProbability("that that"), 0.0000001f);
    }



    @Test
    public void checkDataSparsity(){
        String sentence = "hello world".toLowerCase();

        ng = NGramModel.getNGram(2, sentence);

        ng.babbleNext("hello");
        try {
            ng.babbleNext("world");
            fail("Data is sparse and should throw exception");
        } catch (DataSparsityException dse){
            // pass
        }

    }

    @Test
    public void checkDataSparsityChild(){
        String sentence = "hello world hi".toLowerCase();

        ng = NGramModel.getNGram(3, sentence);

        ng.babbleNext("hello world");
        try {
            ng.babbleNext("hello hi");
            fail("Data is sparse and should throw exception");
        } catch (DataSparsityException dse){
            // pass
        }

    }


    @Disabled
    public void checkDataSmoothing(){
        String sentence = "hello world hi there".toLowerCase();

        ng = NGramModel.getNGram(3, sentence);

        String babble = ng.babbleNext("hi world");

        assertEquals("there", babble);

    }

}