package nlp;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;


//import jgibblda.Coherence;
import cc.mallet.classify.MaxEnt;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.topics.DMROptimizable;
import cc.mallet.topics.tui.DMRLoader;
import cc.mallet.topics.tui.Perplexity;
import cc.mallet.types.*;
import gnu.trove.TIntIntHashMap;

@SuppressWarnings("deprecation")
public class DMRTopicModelXBeta extends LDAHyperXBeta {

    MaxEnt dmrParameters = null;
    int numFeatures;
    int defaultFeatureIndex;
    int save_thrshld = 18500;
    Pipe parameterPipe = null;

    public double[][] alphaCache;
    double[] alphaSumCache;
    public double[][] alphaCache_perp;
    public double[][] inf_alphaCache_perp;



    String typeTopicsFile;

    public DMRTopicModelXBeta(int numberOfTopics) {
        super(numberOfTopics);
    }


    public void updateTheta (){
        for (int doc=0; doc<data.size();doc++){

            //double Kalpha = 0;
            double[] topicCounts = new double [numTopics];

            Arrays.fill(topicCounts, 0);

            FeatureSequence fs = (FeatureSequence) data.get(doc).instance.getData();
            int N = fs.getLength();

            LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            for (int token=0; token < N; token++) {
                topicCounts[ currentDocTopics[token] ] ++;
            }

            double Kalpha=0;
            for (int topic=0;topic<numTopics;topic++){
                Kalpha += alphaCache_perp[doc][topic];
            }
            for (int topic=0;topic<numTopics;topic++){
                train_theta[doc][topic] =(float) ((double)topicCounts[topic] + alphaCache_perp[doc][topic]  ) ;
            }
            train_theta[doc][numTopics]=N+Kalpha;

        }//end theta


    }

    public void inf_updateTheta (){

        for (int doc=0; doc< testData.size();doc++){

            //double Kalpha = 0;
            double[] topicCounts = new double [numTopics];

            Arrays.fill(topicCounts, 0);

            FeatureSequence fs = (FeatureSequence) testData.get(doc).instance.getData();
            int N = fs.getLength();

            LabelSequence topicSequence = (LabelSequence) testData.get(doc).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            for (int token=0; token < N; token++) {
                topicCounts[ currentDocTopics[token] ] ++;
            }

            double Kalpha=0;
            for (int topic=0;topic<numTopics;topic++){
                Kalpha += inf_alphaCache_perp[doc][topic];
            }

            for (int topic=0;topic<numTopics;topic++){
                test_theta[doc][topic] =((double)topicCounts[topic] + inf_alphaCache_perp[doc][topic]  ) /(N + Kalpha  );
            }
            test_theta[doc][numTopics]=N+Kalpha;

        }//end theta


    }


    public void updatePhi (){
        for (int t=0;t< numTypes;t++){
            for (int topic=0; topic<numTopics; topic++){
                phi_p[topic][t] = beta/(tokensPerTopic[topic] +betaSum);
                if (getTypeTopicCounts()[t].containsKey(topic)){
                    phi_p[topic][t] = (getTypeTopicCounts()[t].get(topic) + beta)/(tokensPerTopic[topic] +betaSum);

                } // end if
            }//end topic
        } // end types

    }

    public void estimate (int iterationsThisRound) throws IOException {

        System.out.println("Vocabulary: " + numTypes);
        //loglikelihood
        loglikelihoodArray = new double[iterationsThisRound/50 +1];
        int llCount=0;


        numFeatures = data.get(0).instance.getTargetAlphabet().size() + 1;
        defaultFeatureIndex = numFeatures - 1;

        int numDocs = data.size(); // TODO consider beginning by sub-sampling?

        train_theta = new double[numDocs][numTopics+1];
        phi_p = new double[numTopics][numTypes];
        alphaCache = new double[numDocs][numTopics];
        alphaCache_perp = new double[numDocs][numTopics];
        alphaSumCache = new double[numDocs];

        long startTime = System.currentTimeMillis();
        int maxIteration = iterationsSoFar + iterationsThisRound;

        for ( ; iterationsSoFar <= maxIteration; iterationsSoFar++) {
            long iterationStart = System.currentTimeMillis();

            if (showTopicsInterval != 0 && iterationsSoFar != 0 && iterationsSoFar % showTopicsInterval == 0) {
                System.out.println();
                printTopWords (System.out, wordsPerTopic, false);

            }

            if (iterationsSoFar > burninPeriod && saveStateInterval != 0 && iterationsSoFar % saveStateInterval == 0) {
                //this.printState(new File(stateFilename + '.' + iterationsSoFar + ".gz"));

            }
            //optimizeBeta();
            if (iterationsSoFar > burninPeriod && optimizeInterval != 0 &&
                    iterationsSoFar % optimizeInterval == 0) {
                //printDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
                // Train regression parameters
                //System.out.println("new Beta: " +beta);
                learnParameters();
                optimizeBeta();
                //updateTheta();
                //System.out.print("Optimized");
            }

            if (iterationsSoFar > save_thrshld && optimizeInterval != 0 &&
                    iterationsSoFar % (10) == 0) {
                System.out.print("Saved__");
                updateTheta();
                printDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
                printDocumentTopicsThetas(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar)+"_");
                // Train regression parameters
                //System.out.println("new Beta: " +beta);
                //learnParameters();
                //optimizeBeta();
                //System.out.print("Optimized");
            }



            // Loop over every document in the corpus

            for (int doc = 0; doc < numDocs; doc++) {
                FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance.getData();
                LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;

                if (dmrParameters != null) {
                    // set appropriate Alpha parameters
                    setAlphas(data.get(doc).instance);
                    for (int topic=0; topic < numTopics; topic++) {
                        alphaCache_perp[doc][topic] = alpha[topic];
                        //System.out.println(alpha[topic]);
                    }
                }

                sampleTopicsForOneDoc (tokenSequence, topicSequence,
                        false, false);

            }

            long ms = System.currentTimeMillis() - iterationStart;

            /**if (ms > 1000) {
             System.out.print(Math.round(ms / 1000) + "s ");
             }
             else {
             System.out.print(ms + "ms ");
             }
             */

            if (iterationsSoFar % 50 == 0) {
                System.out.println ("<" + iterationsSoFar + "> ");
                double ll= modelLogLikelihood();
                if (printLogLikelihood) System.out.println ("model ll: "+modelLogLikelihood());
                loglikelihoodArray[llCount] = ll;
                llCount++;
            }
            System.out.flush();
        }


        updatePhi();
        printTypeTopics(typeTopicsFile);
        long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
        long minutes = seconds / 60;    seconds %= 60;
        long hours = minutes / 60;  minutes %= 60;
        long days = hours / 24; hours %= 24;
        System.out.print ("\nTotal time: ");
        if (days != 0) { System.out.print(days); System.out.print(" days "); }
        if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
        if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
        System.out.print(seconds); System.out.println(" seconds");
        updateTheta();
        /**
         for (int doc=0;doc<numDocs;doc++){

         for(int top=0;top<numTopics;top++){
         System.out.println("doc: "+doc+" topic: "+top+" alpha[topic]"+ alphaCache_perp[doc][top]);
         }

         }

         double[] parameters = dmrParameters.getParameters();
         System.out.println("dmr parameter length: "+parameters.length);
         System.out.println("dmr parameter[0]: "+parameters[0]);
         */
    }
    public void infer (int numSamples) throws IOException {

        System.out.println("Vocabulary: " + numTypes);
        //loglikelihood
        loglikelihoodArray = new double[numSamples/50 +1];
        int llCount=0;


        numFeatures = testData.get(0).instance.getTargetAlphabet().size() + 1;
        defaultFeatureIndex = numFeatures - 1;

        int numDocs = testData.size(); // TODO consider beginning by sub-sampling?

        test_theta = new double[numDocs][numTopics+1];
        //phi_p = new double[numTopics][numTypes];
        alphaCache = new double[numDocs][numTopics];
        inf_alphaCache_perp = new double[numDocs][numTopics];
        alphaSumCache = new double[numDocs];

        long startTime = System.currentTimeMillis();
        int maxIteration = iterationsSoFar + numSamples;

        for ( ; iterationsSoFar <= maxIteration; iterationsSoFar++) {
            long iterationStart = System.currentTimeMillis();

            if (showTopicsInterval != 0 && iterationsSoFar != 0 && iterationsSoFar % showTopicsInterval == 0) {
                System.out.println();
                printTopWords (System.out, wordsPerTopic, false);

            }
            /**
             if (iterationsSoFar > burninPeriod && saveStateInterval != 0 && iterationsSoFar % saveStateInterval == 0) {
             //this.printState(new File(stateFilename + '.' + iterationsSoFar + ".gz"));

             }
             //optimizeBeta();

             if (iterationsSoFar > burninPeriod && optimizeInterval != 0 &&
             iterationsSoFar % optimizeInterval == 0) {
             //printDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
             // Train regression parameters
             //System.out.println("new Beta: " +beta);
             //learnParameters();
             //optimizeBeta();
             //updateTheta();
             //System.out.print("Optimized");
             }
             */
            if (iterationsSoFar > save_thrshld && optimizeInterval != 0 &&
                    iterationsSoFar % 10 == 0) {
                inf_updateTheta();
                printShortTestDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
                //printTestDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
                // Train regression parameters
                //System.out.println("new Beta: " +beta);
                //learnParameters();
                //optimizeBeta();
                //System.out.print("Optimized");
            }



            // Loop over every document in the corpus

            for (int doc = 0; doc < numDocs; doc++) {
                FeatureSequence tokenSequence = (FeatureSequence) testData.get(doc).instance.getData();
                LabelSequence topicSequence = (LabelSequence) testData.get(doc).topicSequence;

                if (dmrParameters != null) {
                    // set appropriate Alpha parameters
                    setAlphas(testData.get(doc).instance);
                    for (int topic=0; topic < numTopics; topic++) {
                        inf_alphaCache_perp[doc][topic] = alpha[topic];
                        //System.out.println(alpha[topic]);
                    }
                }

                inferTopicsForOneDoc (tokenSequence, topicSequence,
                        false, false);

            }

            long ms = System.currentTimeMillis() - iterationStart;

            /**if (ms > 1000) {
             System.out.print(Math.round(ms / 1000) + "s ");
             }
             else {
             System.out.print(ms + "ms ");
             }
             */


        }


        //updatePhi();
        printTypeTopics(typeTopicsFile);
        long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
        long minutes = seconds / 60;    seconds %= 60;
        long hours = minutes / 60;  minutes %= 60;
        long days = hours / 24; hours %= 24;
        System.out.print ("\nTotal time: ");
        if (days != 0) { System.out.print(days); System.out.print(" days "); }
        if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
        if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
        System.out.print(seconds); System.out.println(" seconds");
        inf_updateTheta();

    }
    public void infer_old (int numSamples) throws IOException {
        //dmrParameters = null;
        //double[] parameters = dmrParameters.getParameters();
        //System.out.println("dmr parameter length: "+parameters.length);
        //System.out.println("dmr parameter[0]: "+parameters[0]);
        //System.out.println("numFeatures: "+ numFeatures);
        //System.out.println("defaultFeatureIndex: "+ defaultFeatureIndex);

        //numFeatures = data.get(0).instance.getTargetAlphabet().size() + 1;
        //defaultFeatureIndex = numFeatures - 1;

        //numFeatures = data.get(0).instance.getTargetAlphabet().size() + 1;
        //defaultFeatureIndex = numFeatures - 1;

        int numDocs = testData.size(); // TODO consider beginning by sub-sampling?
        System.out.println("Test Docs : " +testData.size());
        test_theta = new double[numDocs][numTopics];
        alphaCache = new double[numDocs][numTopics];
        alphaCache_perp = new double[numDocs][numTopics];
        alphaSumCache = new double[numDocs];

        long startTime = System.currentTimeMillis();
        int maxIteration = iterationsSoFar + numSamples;

        for ( ; iterationsSoFar <= maxIteration; iterationsSoFar++) {
            long iterationStart = System.currentTimeMillis();

            if (showTopicsInterval != 0 && iterationsSoFar != 0 && iterationsSoFar % showTopicsInterval == 0) {
                System.out.println();
                printTopWords (System.out, wordsPerTopic, false);
            }

            if (saveStateInterval != 0 && iterationsSoFar % saveStateInterval == 0) {
                this.printState(new File(stateFilename + '.' + iterationsSoFar + ".gz"));
            }

            if (iterationsSoFar > burninPeriod && optimizeInterval != 0 &&
                    iterationsSoFar % optimizeInterval == 0
                //iterationsSoFar > 0 && optimizeInterval != 0 &&
                //iterationsSoFar % optimizeInterval == 0
                    ) {
                //if (iterationsSoFar > burninPeriod ) {

                //inf_updateTheta();
                //System.out.println("Printing" + xx);
                printTestDocumentTopics(topicProportionOutputFile+"_" +Integer.toString(iterationsSoFar));
                //}
                //testlearnParameters();
                //optimizeBeta();
                // Train regression parameters
                //learnParameters();
            }


            // Loop over every document in the corpus

            for (int doc = 0; doc < numDocs; doc++) {
                FeatureSequence tokenSequence = (FeatureSequence) testData.get(doc).instance.getData();
                LabelSequence topicSequence = (LabelSequence) testData.get(doc).topicSequence;

                if (dmrParameters != null) {
                    // System.out.println("inferr");
                    setAlphas(testData.get(doc).instance);
                    for (int topic=0; topic < numTopics; topic++) {
                        alphaCache_perp[doc][topic] = alpha[topic];
                        //System.out.println(alpha[topic]);
                    }

                }

                /**
                 sampleTopicsForOneDoc (tokenSequence, topicSequence,
                 false, false);
                 */

                inferTopicsForOneDoc (tokenSequence, topicSequence,
                        false, false);

            }


            long ms = System.currentTimeMillis() - iterationStart;
			/*if (ms > 1000) {
				System.out.print(Math.round(ms / 1000) + "s ");
			}
			else {
				System.out.print(ms + "ms ");
			}*/

            if (iterationsSoFar % 10 == 0) {
                System.out.println ("<" + iterationsSoFar + "> ");
                if (printLogLikelihood) System.out.println (modelLogLikelihood());
                long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
                long minutes = seconds / 60;    seconds %= 60;
                long hours = minutes / 60;  minutes %= 60;
                long days = hours / 24; hours %= 24;
                System.out.print ("\nTotal time: ");
                if (days != 0) { System.out.print(days); System.out.print(" days "); }
                if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
                if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
                System.out.print(seconds); System.out.println(" seconds");
            }
            System.out.flush();
        }

        //printTypeTopics("d:/typeTopics");
        long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
        long minutes = seconds / 60;    seconds %= 60;
        long hours = minutes / 60;  minutes %= 60;
        long days = hours / 24; hours %= 24;
        System.out.print ("\nTotal time: ");
        if (days != 0) { System.out.print(days); System.out.print(" days "); }
        if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
        if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
        System.out.print(seconds); System.out.println(" seconds");
        inf_updateTheta();
        //updatePhi();
    }
    /**
     *  Use only the default features to set the topic prior (use no document features)
     */
    public void setAlphas() {

        double[] parameters = dmrParameters.getParameters();

        alphaSum = 0.0;
        smoothingOnlyMass = 0.0;

        // Use only the default features to set the topic prior (use no document features)
        for (int topic=0; topic < numTopics; topic++) {
            alpha[topic] = Math.exp( parameters[ (topic * numFeatures) + defaultFeatureIndex ] );
            alphaSum += alpha[topic];

            smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
            cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

    }

    /** This method sets the alphas for a hypothetical "document" that contains
     *   a single non-default feature.
     */
    public void setAlphas(int featureIndex) {

        double[] parameters = dmrParameters.getParameters();

        alphaSum = 0.0;
        smoothingOnlyMass = 0.0;

        // Use only the default features to set the topic prior (use no document features)
        for (int topic=0; topic < numTopics; topic++) {
            alpha[topic] = Math.exp(parameters[ (topic * numFeatures) + featureIndex ] +
                    parameters[ (topic * numFeatures) + defaultFeatureIndex ] );
            alphaSum += alpha[topic];

            smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
            cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

    }

    /**
     *  Set alpha based on features in an instance
     */
    public void setAlphas(Instance instance) {

        // we can't use the standard score functions from MaxEnt,
        //  since our features are currently in the Target.
        FeatureVector features = (FeatureVector) instance.getTarget();
        //System.out.println("numFeaturesLocations: "+features.numLocations());
        //System.out.println("feature at location: "+features.indexAtLocation(features.numLocations()-1));
        /**
         if (features == null) {
         System.out.println("Warning: No Features Detected");
         setAlphas(); return; }
         */
        double[] parameters = dmrParameters.getParameters();

        alphaSum = 0.0;
        smoothingOnlyMass = 0.0;
        /**
         * numFeatures = 7
         * defaultFeatureIndex = 6
         *
         */
        //System.out.println(numFeatures);
        //System.out.println(defaultFeatureIndex);
        //System.out.println(parameters[2*numFeatures + defaultFeatureIndex]);
        for (int topic = 0; topic < numTopics; topic++) {
            alpha[topic] = parameters[topic*numFeatures + defaultFeatureIndex] // this is the constant
                    + MatrixOps.rowDotProduct (parameters, // vector product
                    numFeatures,
                    topic, features,
                    defaultFeatureIndex,
                    null); // vector product

            alpha[topic] = Math.exp(alpha[topic]);

            alphaSum += alpha[topic];

            smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum); // eqn 7 Constant for all documents
            cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum); //not 8 or 9
        }
    }

    public void learnParameters() {

        // Create a "fake" pipe with the features in the data and
        //  a trove int-int hashmap of topic counts in the target.

        if (parameterPipe == null) {
            parameterPipe = new Noop();

            parameterPipe.setDataAlphabet(data.get(0).instance.getTargetAlphabet());
            parameterPipe.setTargetAlphabet(topicAlphabet);
        }

        InstanceList parameterInstances = new InstanceList(parameterPipe);

        if (dmrParameters == null) {
            dmrParameters = new MaxEnt(parameterPipe, new double[numFeatures * numTopics]);
        }

        for (int doc=0; doc < data.size(); doc++) {

            if (data.get(doc).instance.getTarget() == null) {
                continue;
            }

            FeatureCounter counter = new FeatureCounter(topicAlphabet);

            for (int topic : data.get(doc).topicSequence.getFeatures()) {
                counter.increment(topic);
            }

            // Put the real target in the data field, and the
            //  topic counts in the target field
            parameterInstances.add( new Instance(data.get(doc).instance.getTarget(), counter.toFeatureVector(), null, null) );

        }

        DMROptimizable optimizable = new DMROptimizable(parameterInstances, dmrParameters);
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
        dmrParameters = optimizable.getClassifier();

        for (int doc=0; doc < data.size(); doc++) {
            Instance instance = data.get(doc).instance;
            FeatureSequence tokens = (FeatureSequence) instance.getData();
            if (instance.getTarget() == null) { continue; }
            int numTokens = tokens.getLength();

            // This sets alpha[] and alphaSum
            setAlphas(instance);

            // Now cache alpha values
            for (int topic=0; topic < numTopics; topic++) {
                alphaCache[doc][topic] = alpha[topic];
            }
            alphaSumCache[doc] = alphaSum;
        }

    }
    public void testlearnParameters() {

        // Create a "fake" pipe with the features in the data and
        //  a trove int-int hashmap of topic counts in the target.

        if (parameterPipe == null) {
            parameterPipe = new Noop();

            parameterPipe.setDataAlphabet(data.get(0).instance.getTargetAlphabet());
            parameterPipe.setTargetAlphabet(topicAlphabet);
        }

        InstanceList parameterInstances = new InstanceList(parameterPipe);

        if (dmrParameters == null) {
            dmrParameters = new MaxEnt(parameterPipe, new double[numFeatures * numTopics]);
        }

        for (int doc=0; doc < testData.size(); doc++) {

            if (testData.get(doc).instance.getTarget() == null) {
                continue;
            }

            FeatureCounter counter = new FeatureCounter(topicAlphabet);

            for (int topic : testData.get(doc).topicSequence.getFeatures()) {
                counter.increment(topic);
            }

            // Put the real target in the data field, and the
            //  topic counts in the target field
            parameterInstances.add( new Instance(testData.get(doc).instance.getTarget(), counter.toFeatureVector(), null, null) );

        }

        DMROptimizable optimizable = new DMROptimizable(parameterInstances, dmrParameters);
        optimizable.setRegularGaussianPriorVariance(0.5);
        optimizable.setInterceptGaussianPriorVariance(100.0);

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
        dmrParameters = optimizable.getClassifier();

        for (int doc=0; doc < testData.size(); doc++) {
            Instance instance = testData.get(doc).instance;
            FeatureSequence tokens = (FeatureSequence) instance.getData();
            if (instance.getTarget() == null) { continue; }
            int numTokens = tokens.getLength();

            // This sets alpha[] and alphaSum
            setAlphas(instance);

            // Now cache alpha values
            for (int topic=0; topic < numTopics; topic++) {
                alphaCache[doc][topic] = alpha[topic];
            }
            alphaSumCache[doc] = alphaSum;
        }

    }
    public void printTopWords (PrintStream out, int numWords, boolean usingNewLines) {
        if (dmrParameters != null) { setAlphas(); }
        super.printTopWords(out, numWords, usingNewLines);
    }

    public void writeParameters(File parameterFile) throws IOException {
        if (dmrParameters != null) {
            PrintStream out = new PrintStream(parameterFile);
            dmrParameters.print(out);
            out.close();
        }
    }

    private static final long serialVersionUID = 1;
    private static final int CURRENT_SERIAL_VERSION = 0;
    private static final int NULL_INTEGER = -1;






    public static void main (String[] args) throws IOException {


        //Controls
        int pseudoCount = 5;
        int Seed = 10;
        int burnin = 1500;
        int optimInterval = 100;
        int numIters = 15000 ;
        int inferIters = 1500;
        String subDir="matched/";
        // nocolon/, short/

        System.out.println(
                "type = " + subDir + '\n' +
                        "pseudoCount = " + pseudoCount + '\n' +
                        "Seed = " + Seed + '\n' +
                        "burnin = " + burnin + '\n' +
                        "optimInterval = " + optimInterval + '\n' +
                        "numIters = " + numIters + '\n' +
                        "inferIters = " + inferIters + '\n'
        );


        String processed_data="D:/Dinesh/Projects/CALORIES/PROCESSED_DATA";





        String subDir_out = subDir + "unseed_";

        if (pseudoCount>0){
            subDir_out = subDir + "seed_";
        }


        String apath ="D:/Dinesh/Projects/CALORIES/topicModelResults/chain/"+subDir_out+Integer.toString(Seed);
        new File(apath).mkdirs();
        File wordsFile = new File(processed_data+"/chain/"+subDir+"chain_text.txt");
        File featuresFile = new File(processed_data+"/chain/"+subDir+"chain_features.txt");
        File instancesFile = new File(processed_data+"/chain/"+subDir+"chain_instances.txt");

        DMRLoader loader = new DMRLoader();
        loader.load(wordsFile, featuresFile, instancesFile);
        System.out.println("Done Training Load");


        String apath_t ="D:/Dinesh/Projects/CALORIES/topicModelResults/non_chain/"+subDir_out+Integer.toString(Seed);
        new File(apath_t).mkdirs();
        File wordsFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_text.txt");
        File featuresFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_features.txt");
        File instancesFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_instances.txt");

        DMRLoader loader_t = new DMRLoader();
        loader_t.load(wordsFile_t, featuresFile_t, instancesFile_t);
        System.out.println("Done Testing Load");




        String[] argss = new String[3];
        argss[0] = processed_data+"/chain/"+subDir+"chain_instances.txt" ;
        //argss[0] = "D:/Dinesh/Projects/CALORIES/PROCESSED_DATA/chain/plain_chain_instances.txt" ;
        argss[1] = "200" ;
        argss[2] = processed_data+"/non_chain/"+subDir+"nonchain_instances.txt" ;

        InstanceList training = InstanceList.load (new File(argss[0]));
        //InstanceList training = InstanceList.load (new File("D:/texts.txt"));

        int numTopics = argss.length > 1 ? Integer.parseInt(argss[1]) : 20;
        //int numTopics = 100;

        InstanceList testing =
                argss.length > 2 ? InstanceList.load (new File(argss[2])) : null;

        DMRTopicModelXBeta lda = new DMRTopicModelXBeta (numTopics);
        lda.setSeed(Seed);
        //lda.defaultFeatureIndex=0;
        lda.setBurninPeriod(burnin);
        lda.setOptimizeInterval(optimInterval);
        lda.saveStateInterval = 100;
        lda.printLogLikelihood = true;
        lda.setTopicDisplay(100, 10);
        lda.setPseudoCount(pseudoCount);
        lda.setSeededTopics(Arrays.asList("calories","carbohydrates","fat","protein","vitamin","cholestrol","calcium","magnesium","sodium","potassium","iron","iodine" ));
        lda.typeTopicsFile = apath+"/typeTopics";
        lda.addInstances(training);
        lda.setNumIterations(numIters);
        lda.save_thrshld =numIters-600 ;
        lda.setTopicProportionFile(apath+"/topicProportions");
        lda.beta = 0.01; //never change
        //lda.dmrParameters.setParameter(0, 0, 0);
        lda.estimate();
        lda.writeParameters(new File(apath+"/dmr.parameters"));
        lda.printState(new File(apath+"/dmr.state.gz"));

        Coherence cohere = new Coherence();

        cohere.collectDocumentStatistics( lda.numTopics, 20, lda.numTypes,
                lda.getTypeTopicCounts() ,lda, apath, "train");
        //Perplexity perp = new Perplexity();
        Perplexity.calc(lda);
        //Inference
        lda.save_thrshld = inferIters-600 ;
        lda.iterationsSoFar = 0;
        lda.typeTopicsFile = apath_t+"/typeTopics";
        lda.setTopicProportionFile(apath_t+"/topicProportions");

        lda.addTestInstances(testing);
        lda.infer(inferIters);
        lda.writeParameters(new File(apath_t+"/dmr.parameters"));
        lda.printState(new File(apath_t+"/dmr.state.gz"));
        Perplexity.testCalc(lda);


    }

    public List<Topication> data() {
        return data;
    }

    public Alphabet alphabet() {
        return alphabet;
    }

    public LabelAlphabet topicAlphabet() {
        return topicAlphabet;
    }

    public TIntIntHashMap[] getTypeTopicCounts() {
        return typeTopicCounts;
    }

    public int numTopics() {
        return numTopics;
    }

    public double[] alpha() {
        return alpha;
    }

    public double beta() {
        return beta;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public int numTypes() {
        return numTypes;
    }

    public void setSaveStateInterval(int saveStateInterval) {
        this.saveStateInterval = saveStateInterval;
    }

    public void setPrintLogLikelihood(boolean printLogLikelihood) {
        this.printLogLikelihood = printLogLikelihood;
    }


}
