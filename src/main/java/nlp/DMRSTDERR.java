package nlp;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
//import java.util.List;
import java.util.Random;

import au.com.bytecode.opencsv.CSVWriter;
import cc.mallet.classify.MaxEnt;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.pipe.Noop;
import cc.mallet.types.FeatureCounter;

import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;


/**
 * @author theDesktop
 *
 */

@SuppressWarnings("deprecation")
public class DMRSTDERR {

    /**
     * Note this class only save the coefficeints for lambda
     */
    int sampleSize;
    Random rg = new Random(5);
    double[] parameters_orig;
    double[][] health_topic; // This is D x M in size where there are D documents and M coefficients
    //MaxEnt orig_dmr;
    ArrayList<Integer> doc_list;

    public void unpacker(int iter , DMRTopicModelXBeta model ){
        /**
         * Convert dmr parameters for health topic
         */
        double[] parameters = model.dmrParameters.getParameters();
        //int topic=0;
        //System.out.println(model.defaultFeatureIndex+" : "+model.numFeatures+":"+parameters[model.defaultFeatureIndex]);
        for (int topic = 0; topic < 1; topic++) {
            this.health_topic[iter][0] = parameters[topic*model.numFeatures + model.defaultFeatureIndex];
            System.out.println(health_topic[iter][0] );
            int size = model.numFeatures;
            for (int cil = 0; cil < size-1; cil++) {
                this.health_topic[iter][cil+1] = parameters[topic*model.numFeatures+cil];

            }


        }
        System.out.println("unpacked" );
    }// end unpacker


    public void learnParameters(int sample , DMRTopicModelXBeta model) {

        // Create a "fake" pipe with the features in the data and
        //  a trove int-int hashmap of topic counts in the target.

        ArrayList<Integer> subSample = sample(doc_list, sampleSize);

        if (model.parameterPipe == null) {
            model.parameterPipe = new Noop();

            model.parameterPipe.setDataAlphabet(model.data.get(0).instance.getTargetAlphabet());
            model.parameterPipe.setTargetAlphabet(model.topicAlphabet);
        }

        InstanceList parameterInstances = new InstanceList(model.parameterPipe);

        if (model.dmrParameters == null) {
            model.dmrParameters = new MaxEnt(model.parameterPipe, new double[model.numFeatures * model.numTopics]);
        }



        for (int doc : subSample) {

            if (model.data.get(doc).instance.getTarget() == null) {
                continue;
            }

            FeatureCounter counter = new FeatureCounter(model.topicAlphabet);

            for (int topic : model.data.get(doc).topicSequence.getFeatures()) {
                counter.increment(topic);
            }

            // Put the real target in the data field, and the
            //  topic counts in the target field
            parameterInstances.add( new Instance(model.data.get(doc).instance.getTarget(), counter.toFeatureVector(), null, null) );

        }

        DMROptimizable optimizable = new DMROptimizable(parameterInstances, model.dmrParameters);
        optimizable.setRegularGaussianPriorVariance(0.5);
        optimizable.setInterceptGaussianPriorVariance(100.0);
        //optimizable.setRegularGaussianPriorVariance(5);
        //optimizable.setInterceptGaussianPriorVariance(5);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

        // Optimize once
        try {
            optimizer.optimize();
        } catch (OptimizationException e) {
            // step size too small
        }

        // Restart with a fresh initialization to improve likelihood
        try {
            optimizer.optimize();
        } catch (OptimizationException e) {
            // step size too small
        }
        model.dmrParameters = optimizable.getClassifier(); //what is the size of this? K x M?
        System.out.println(model.dmrParameters.getParameters()[model.defaultFeatureIndex] );
        unpacker(sample,model);

        //reset to original point estimate
        model.dmrParameters.setParameters(parameters_orig.clone());
        System.out.println(model.dmrParameters.getParameters()[model.defaultFeatureIndex] );

    }// end learn parameters

    public void save_lambdas(String fileName,int numFeatures)throws IOException	{
        /**
         * Save Lambdas to file say csv
         *
         */
        CSVWriter bwriter  = new CSVWriter(new FileWriter(fileName+".csv"), ',');
        String[] header = new String[1+numFeatures] ;
        header[0] ="docID";


        for (int features = 0; features < numFeatures; features++) {
            header[features+1] = "f_" + Integer.toString(features) ;

        }
        bwriter.writeNext(header);


        for (int di = 0; di < health_topic.length; di++) {

            String []  stringArray = new String[numFeatures+1] ;
            stringArray[0] =Integer.toString( di);

            for (int i = 0; i < numFeatures; i++) {
                stringArray[i+1]=String.valueOf(health_topic[di][i]) ;
            }

            bwriter.writeNext(stringArray);

        }
        bwriter.close();

    }

    public ArrayList<Integer> sample(ArrayList<Integer> set, int size) {


        ArrayList<Integer> out = new ArrayList<Integer>();
        while (out.size() < size) {
            out.add(set.get(rg.nextInt(set.size())));
        }
        return out;
    }


    public void standard_err(DMRTopicModelXBeta model , String fileName) throws IOException{

        /**
         * Accepts a set of instances, current alpha cache
         * beta and then creates an optimizable
         * 1. run a loop for 1000 samples
         * 2. in each loop sample with replacement 1000 documents
         * 3. call optimizable (on all topics)
         * 4. save the lambda coefficients in a matrix.
         * 5. write the coefficients to a file
         * returns nothing
         */


        sampleSize = model.data.size();//
        int numSamples= 1000;
        parameters_orig = model.dmrParameters.getParameters().clone();
        //orig_dmr = model.dmrParameters.setParameters(parameters);;
        this.health_topic = new double[numSamples][model.numFeatures];

        unpacker(0,model);
        this.doc_list = new ArrayList<Integer>();

        for (int doc = 0; doc < model.data.size(); doc++) {
            this.doc_list.add(doc);
        }


        for (int sample = 1; sample < numSamples; sample++) {
            learnParameters(sample, model);
        }

        save_lambdas(fileName, model.numFeatures);
        System.out.println("Stderr Done");


    }



}
