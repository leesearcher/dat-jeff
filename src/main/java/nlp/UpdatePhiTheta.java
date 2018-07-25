package cc.mallet.topics;

import java.util.Arrays;

import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;
import nlp.DMRTopicModelXBeta;

public class UpdatePhiTheta {

    @SuppressWarnings("deprecation")
    public static double[][] updateTheta (DMRTopicModelXBeta model){
        double[][] theta = new double[model.data.size()][model.numTopics];
        int numTopics = model.numTopics;
        int numTypes = model.getTypeTopicCounts().length;
        System.out.println("numTypes " + numTypes);
        System.out.println("trainData " + model.data.size());
        System.out.println("Cache length = "+model.alphaCache.length);
        for (int doc=0; doc<model.data.size();doc++){

            double Kalpha = 0;
            double[] topicCounts = new double [numTopics];

            Arrays.fill(topicCounts, 0);

            FeatureSequence fs = (FeatureSequence) model.data.get(doc).instance.getData();
            int N = fs.getLength();

            LabelSequence topicSequence = (LabelSequence) model.data.get(doc).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            for (int token=0; token < N; token++) {
                topicCounts[ currentDocTopics[token] ] ++;
            }

            for (int topic=0;topic<model.numTopics;topic++){
                Kalpha += model.alphaCache[doc][topic];
            }
            for (int topic=0;topic<model.numTopics;topic++){
                theta[doc][topic] =(float) (topicCounts[topic] + model.alpha[topic]  )/(N+Kalpha );
            }
        }//end theta



        return theta;

    }


    @SuppressWarnings("deprecation")
    public static double[][] updatePhi (DMRTopicModelXBeta model){
        double beta = model.beta;
        int numTopics = model.numTopics;
        int numTypes = model.getTypeTopicCounts().length;
        System.out.println("numTypes " + numTypes);

        double[][] phi = new double[numTopics][numTypes]; //K x V

        System.out.println("Cache length = "+model.alphaCache.length);



        for (int t=0;t< numTypes;t++){
            for (int topic=0; topic<numTopics; topic++){
                phi[topic][t] = 0;
                if (model.getTypeTopicCounts()[t].containsKey(topic)){
                    phi[topic][t] = model.getTypeTopicCounts()[t].get(topic) + beta;
                } // end if
            }//end topic
        } // end types
        return phi;

    }



    @SuppressWarnings("deprecation")
    public double[][] inf_updateTheta (DMRTopicModelXBeta model){
        /**
         * Must only be run after inference
         */
        double[][] theta = new double[model.testData.size()][model.numTopics];
        int numTopics = model.numTopics;
        int numTypes = model.getTypeTopicCounts().length;
        System.out.println("numTypes " + numTypes);
        System.out.println("trainData " + model.testData.size());
        System.out.println("Cache length = "+model.alphaCache.length);
        for (int doc=0; doc<model.testData.size();doc++){

            double Kalpha = 0;
            double[] topicCounts = new double [numTopics];

            Arrays.fill(topicCounts, 0);

            FeatureSequence fs = (FeatureSequence) model.testData.get(doc).instance.getData();
            int N = fs.getLength();

            LabelSequence topicSequence = (LabelSequence) model.testData.get(doc).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            for (int token=0; token < N; token++) {
                topicCounts[ currentDocTopics[token] ] ++;
            }

            for (int topic=0;topic<model.numTopics;topic++){
                Kalpha += model.alphaCache[doc][topic];
            }
            for (int topic=0;topic<model.numTopics;topic++){
                theta[doc][topic] =(float) (topicCounts[topic] + model.alpha[topic]  )/(N+Kalpha );
            }
        }//end theta



        return theta;

    }



}
