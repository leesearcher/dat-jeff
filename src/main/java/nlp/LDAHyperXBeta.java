package nlp;

import cc.mallet.topics.LDAHyper;
import cc.mallet.types.*;
import gnu.trove.TIntIntHashMap;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import au.com.bytecode.opencsv.CSVWriter;
import cc.mallet.topics.LDAHyper.Topication;
import cc.mallet.util.Randoms;




@SuppressWarnings("deprecation")
public class LDAHyperXBeta extends LDAHyper {



    /**
     *
     */
    private static final long serialVersionUID = 1L;


    protected int pseudoCount = 0;
    protected List<String> seedWords;
    protected String topicProportionOutputFile;
    public ArrayList<Topication> testData = new ArrayList<Topication>();

    //for beta optimization
    protected int[] typeTotals;
    protected int maxTypeCount;
    protected double[] loglikelihoodArray ;
    public double [][] train_theta;
    public double [][] test_theta;
    public double [][] phi_p;

    public LDAHyperXBeta (int numberOfTopics) {
        super(numberOfTopics);
    }

    public void setPseudoCount(int seed) {
        this.pseudoCount = seed;
    }
    public void setSeededTopics(List<String> tops){
        this.seedWords =tops;
    }

    public void setTopicProportionFile(String topicProportionOutputFile){
        this.topicProportionOutputFile =topicProportionOutputFile;
    }


    protected void initializeHistogramsAndCachedValues() {


        for (String item : seedWords){
            if(alphabet.contains(item)==true){
                int typeID = alphabet.lookupIndex(item);
                //System.out.println(item +" : " +typeID);
                //System.out.println(item+" : "+alphabet.lookupObject(typeID));

                tokensPerTopic[0] +=pseudoCount;
                typeTopicCounts[typeID].adjustOrPutValue(0, pseudoCount, pseudoCount);
            }
            //typeTopicCounts[typeID].adjustOrPutValue(0, pseudoCount, pseudoCount);
        }

        int maxTokens = 0;
        int totalTokens = 0;
        int seqLen;

        for (int doc = 0; doc < data.size(); doc++) {
            FeatureSequence fs = (FeatureSequence) data.get(doc).instance.getData();
            seqLen = fs.getLength();
            if (seqLen > maxTokens)
                maxTokens = seqLen;
            totalTokens += seqLen;
        }
        // Initialize the smoothing-only sampling bucket
        smoothingOnlyMass = 0;
        for (int topic = 0; topic < numTopics; topic++)
            smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);

        // Initialize the cached coefficients, using only smoothing.
        cachedCoefficients = new double[ numTopics ];
        for (int topic=0; topic < numTopics; topic++)
            cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);

        System.err.println("max tokens: " + maxTokens);
        System.err.println("total tokens: " + totalTokens);

        docLengthCounts = new int[maxTokens + 1];
        topicDocCounts = new int[numTopics][maxTokens + 1];
    }




    public void printDocumentTopicsThetas (String filename) throws IOException	{
        System.out.println("Printing Thetas");
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+"_theta.csv"), ',');
        String[] header = new String[1+numTopics] ;
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        //int docLen;
        int[] topicCounts = new int[ numTopics ];


        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            //sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +data.size());

        for (int di = 0; di < data.size(); di++) {
            String []  stringArray = new String[numTopics+1] ;
            stringArray[0] =Integer.toString( di);
            double [] sortedTopics = new double[ numTopics ];
            // And normalize
            for (int topic = 0; topic < numTopics; topic++) {
                //sortedTopics[topic].set(topic, (float) topicCounts[topic] / docLen);
                sortedTopics[topic]=  train_theta[di][topic];
            }

            //Arrays.sort(sortedTopics);

            for (int i = 0; i < max; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i]) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            //Arrays.fill(topicCounts, 0);
        }
        bwriter.close();
        //printDocumentTopicsRaw(filename);
    }



    public void printDocumentTopics (String filename) throws IOException	{
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+".csv"), ',');
        String[] header = new String[1+numTopics] ;
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        int docLen;
        int[] topicCounts = new int[ numTopics ];

        IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +data.size());

        for (int di = 0; di < data.size(); di++) {
            //System.out.println("Doc processed : " +di);
            LabelSequence topicSequence = (LabelSequence) data.get(di).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            //pw.print (di); pw.print (' ');
            String []  stringArray = new String[numTopics+1] ;
            stringArray[0] =Integer.toString( di);
            //String jjj=String.valueOf(di) + " "
            //bwriter.writeNext(jjj);

			/*if (data.get(di).instance.getSource() != null) {
				//pw.print (data.get(di).instance.getSource());
			}
			else {
				//pw.print ("null-source");
			}*/

            //pw.print (' ');
            docLen = currentDocTopics.length;
            //System.out.println("Doc len: " +docLen);
            // Count up the tokens
            for (int token=0; token < docLen; token++) {
                topicCounts[ currentDocTopics[token] ]++;
            }


            // And normalize
            for (int topic = 0; topic < numTopics; topic++) {
                sortedTopics[topic].set(topic, (float) topicCounts[topic] / docLen);
                //sortedTopics[topic].set(topic, (float) train_theta[di][topic]);
            }

            //Arrays.sort(sortedTopics);

            for (int i = 0; i < max; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i].getWeight()) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            Arrays.fill(topicCounts, 0);
        }
        bwriter.close();
        //printDocumentTopicsRaw(filename);
    }

    public void printDocumentTopicsRaw (String filename) throws IOException	{
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+"_raw.csv"), ',');
        String[] header = new String[1+numTopics] ;
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        int docLen;
        int[] topicCounts = new int[ numTopics ];

        IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +data.size());

        for (int di = 0; di < data.size(); di++) {
            //System.out.println("Doc processed : " +di);
            LabelSequence topicSequence = (LabelSequence) data.get(di).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            //pw.print (di); pw.print (' ');
            String []  stringArray = new String[numTopics+1] ;
            stringArray[0] =Integer.toString( di);
            //String jjj=String.valueOf(di) + " "
            //bwriter.writeNext(jjj);

			/*if (data.get(di).instance.getSource() != null) {
				//pw.print (data.get(di).instance.getSource());
			}
			else {
				//pw.print ("null-source");
			}*/

            //pw.print (' ');
            docLen = currentDocTopics.length;
            //System.out.println("Doc len: " +docLen);
            // Count up the tokens
            for (int token=0; token < docLen; token++) {
                topicCounts[ currentDocTopics[token] ]++;
            }

            // And normalize
            for (int topic = 0; topic < numTopics; topic++) {
                sortedTopics[topic].set(topic,  topicCounts[topic] );
            }

            //Arrays.sort(sortedTopics);

            for (int i = 0; i < max; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i].getWeight()) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            Arrays.fill(topicCounts, 0);
        }
        bwriter.close();
    }

    public void printTypeTopics (String filename) throws IOException	{
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+".csv"), ',');
        String[] header = new String[2+numTopics] ;
        header[0] ="Type";
        header[1] ="TypeID";

        //pw.print ("#doc source topic proportion ...\n");

        //IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+2] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            //sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        for (int t = 0; t < numTypes; t++) {
            String []  stringArray = new String[numTopics+2] ;
            stringArray[0]=Integer.toString(t);
            stringArray[1]=(String) alphabet.lookupObject(t);
            for (int topic = 0; topic < numTopics; topic++) {
                stringArray[topic+2] = Double.toString(phi_p[topic][t]);
                //stringArray[topic+2] = Integer.toString(getTypeTopicCounts()[t].get(topic));
                //sortedTopics[topic].set(topic, (float) topicCounts[topic] / docLen);
            }
            bwriter.writeNext(stringArray);
        }


        bwriter.close();


        CSVWriter cwriter  = new CSVWriter(new FileWriter(filename+"_ll.csv"), ',');
        for (int ll =0; ll<loglikelihoodArray.length; ll++){
            String []  stringArray = new String[1] ;
            stringArray[0] =Double.toString(loglikelihoodArray[ll]);
            cwriter.writeNext(stringArray);
        }
        cwriter.close();


        //model params
        CSVWriter pwriter  =  new CSVWriter(new FileWriter(filename+"_params.csv"), ',');
        int[] pars = {pseudoCount, numIterations,burninPeriod , optimizeInterval};
        String[] parsName = {"pseudoCount", "numIterations","burninPeriod" , "optimizeInterval"};
        for (int par = 0; par < pars.length; par++){
            String []  stringArray = new String[2] ;
            stringArray[0] =parsName[par] ;
            stringArray[1] = Integer.toString(pars[par]);
            pwriter.writeNext(stringArray);
        }
        pwriter.close();

    }

    public void addInstances (InstanceList training) {
        initializeForTypes(training.getDataAlphabet());
        ArrayList<LabelSequence> topicSequences = new ArrayList<LabelSequence>();
        for (Instance instance : training) {
            LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[instanceLength(instance)]);
            //Randoms r = new Randoms();
            Randoms r = random;
            int[] topics = topicSequence.getFeatures();
            for (int i = 0; i < topics.length; i++)
                topics[i] = r.nextInt(numTopics);
            topicSequences.add (topicSequence);
        }
        addInstances (training, topicSequences);
    }


    public void addInstances (InstanceList training, List<LabelSequence> topics) {
        initializeForTypes (training.getDataAlphabet());
        assert (training.size() == topics.size());
        typeTotals = new int[numTypes];
        for (int i = 0; i < training.size(); i++) {
            Topication t = new Topication (training.get(i), this, topics.get(i));
            data.add (t);
            // Include sufficient statistics for this one doc
            FeatureSequence tokenSequence = (FeatureSequence) t.instance.getData();
            LabelSequence topicSequence = t.topicSequence;
            for (int pi = 0; pi < topicSequence.getLength(); pi++) {
                int topic = topicSequence.getIndexAtPosition(pi);
                typeTopicCounts[tokenSequence.getIndexAtPosition(pi)].adjustOrPutValue(topic, 1, 1);
                tokensPerTopic[topic]++;
                typeTotals[tokenSequence.getIndexAtPosition(pi)]++;
            }
        }
        initializeHistogramsAndCachedValues();
        maxTypeCount = 0;
        for (int type = 0; type < numTypes; type++) {
            if (typeTotals[type] > maxTypeCount) { maxTypeCount = typeTotals[type]; }
            //typeTopicCounts[type] = new int[ Math.min(numTopics, typeTotals[type]) ];
        }
    }



    public void addTestInstances (InstanceList testing) {
        //initializeForTypes (training.getDataAlphabet());
        ArrayList<LabelSequence> topicSequences = new ArrayList<LabelSequence>();
        for (Instance instance : testing) {
            LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[instanceLength(instance)]);
            //Randoms r = new Randoms(5);

            Randoms r = random;
            //r.setSeed(5);
            int[] topics = topicSequence.getFeatures();
            for (int i = 0; i < topics.length; i++)
                topics[i] = r.nextInt(numTopics);
            topicSequences.add (topicSequence);
        }
        addTestInstances(testing, topicSequences);

    }

    public void addTestInstances(InstanceList testing, List<LabelSequence> topics) {
        //initializeForTypes (training.getDataAlphabet());
        assert (testing.size() == topics.size());
        for (int i = 0; i < testing.size(); i++) {
            Topication t = new Topication (testing.get(i), this, topics.get(i));
            testData.add (t);
            // Include sufficient statistics for this one doc
            /**
             FeatureSequence tokenSequence = (FeatureSequence) t.instance.getData();
             LabelSequence topicSequence = t.topicSequence;
             for (int pi = 0; pi < topicSequence.getLength(); pi++) {
             int topic = topicSequence.getIndexAtPosition(pi);
             //typeTopicCounts[tokenSequence.getIndexAtPosition(pi)].adjustOrPutValue(topic, 1, 1);
             //tokensPerTopic[topic]++;*/
        }
    }


    public void printTestDocumentTopics (String filename) throws IOException	{
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+".csv"), ',');
        String[] header = new String[1+numTopics] ;
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        int docLen;
        int[] topicCounts = new int[ numTopics ];

        IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +testData.size());

        for (int di = 0; di < testData.size(); di++) {
            //System.out.println("Doc processed : " +di);
            LabelSequence topicSequence = (LabelSequence) testData.get(di).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            //pw.print (di); pw.print (' ');
            String []  stringArray = new String[numTopics+1] ;
            stringArray[0] =Integer.toString( di);
            //String jjj=String.valueOf(di) + " "
            //bwriter.writeNext(jjj);

			/*if (data.get(di).instance.getSource() != null) {
				//pw.print (data.get(di).instance.getSource());
			}
			else {
				//pw.print ("null-source");
			}*/

            //pw.print (' ');
            docLen = currentDocTopics.length;
            //System.out.println("Doc len: " +docLen);
            // Count up the tokens
            for (int token=0; token < docLen; token++) {
                topicCounts[ currentDocTopics[token] ]++;
            }

            // And normalize
            for (int topic = 0; topic < numTopics; topic++) {
                sortedTopics[topic].set(topic, (float) test_theta[di][topic]);
            }

            //Arrays.sort(sortedTopics);

            for (int i = 0; i < max; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i].getWeight()) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            Arrays.fill(topicCounts, 0);
        }
        bwriter.close();
        //printTestDocumentTopicsRaw(filename);
    }
    public void printShortTestDocumentTopics (String filename) throws IOException	{
        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+".csv"), ',');
        String[] header = new String[1+numTopics] ;//docid and health topic only - otherwise fiule size is too big.
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        int docLen;
        int[] topicCounts = new int[ numTopics ];

        IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +testData.size());

        for (int di = 0; di < testData.size(); di++) {
            //System.out.println("Doc processed : " +di);
            //LabelSequence topicSequence = (LabelSequence) testData.get(di).topicSequence;
            //int[] currentDocTopics = topicSequence.getFeatures();

            //pw.print (di); pw.print (' ');
            String []  stringArray = new String[2+1] ;
            stringArray[0] =Integer.toString( di);
            //String jjj=String.valueOf(di) + " "
            //bwriter.writeNext(jjj);

			/*if (data.get(di).instance.getSource() != null) {
				//pw.print (data.get(di).instance.getSource());
			}
			else {
				//pw.print ("null-source");
			}*/

            //pw.print (' ');
            //docLen = currentDocTopics.length;
            //System.out.println("Doc len: " +docLen);
            // Count up the tokens

            sortedTopics[0].set(0, (float) test_theta[di][0]);
            sortedTopics[1].set(0, (float) test_theta[di][max]);



            //Arrays.sort(sortedTopics);

            for (int i = 0; i <2; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i].getWeight()) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            Arrays.fill(topicCounts, 0);
        }
        bwriter.close();
        //printTestDocumentTopicsRaw(filename);
    }
    public void printTestDocumentTopicsRaw (String filename) throws IOException	{


        //BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filename)), "UTF-8"));
        CSVWriter bwriter  = new CSVWriter(new FileWriter(filename+"_raw.csv"), ',');
        String[] header = new String[1+numTopics] ;
        header[0] ="docID";
        //pw.print ("#doc source topic proportion ...\n");
        int docLen;
        int[] topicCounts = new int[ numTopics ];

        IDSorter[] sortedTopics = new IDSorter[ numTopics ];
        for (int topic = 0; topic < numTopics; topic++) {
            header[topic+1] = "topic_" + Integer.toString(topic) ;
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }
        bwriter.writeNext(header);
        int max = numTopics;
		/*if (max < 0 || max > numTopics) {
			max = numTopics;
		}*/
        System.out.println("Topics : " +max );
        System.out.println("Docs : " +testData.size());

        for (int di = 0; di < testData.size(); di++) {
            //System.out.println("Doc processed : " +di);
            LabelSequence topicSequence = (LabelSequence) testData.get(di).topicSequence;
            int[] currentDocTopics = topicSequence.getFeatures();

            //pw.print (di); pw.print (' ');
            String []  stringArray = new String[numTopics+1] ;
            stringArray[0] =Integer.toString( di);
            //String jjj=String.valueOf(di) + " "
            //bwriter.writeNext(jjj);

			/*if (data.get(di).instance.getSource() != null) {
				//pw.print (data.get(di).instance.getSource());
			}
			else {
				//pw.print ("null-source");
			}*/

            //pw.print (' ');
            docLen = currentDocTopics.length;
            //System.out.println("Doc len: " +docLen);
            // Count up the tokens
            for (int token=0; token < docLen; token++) {
                topicCounts[ currentDocTopics[token] ]++;
            }

            // And normalize
            for (int topic = 0; topic < numTopics; topic++) {
                sortedTopics[topic].set(topic,  topicCounts[topic] );
            }

            //Arrays.sort(sortedTopics);

            for (int i = 0; i < max; i++) {
                //if (sortedTopics[i].getWeight() < threshold) { break; }

                //pw.print (sortedTopics[i].getID() + " " + sortedTopics[i].getWeight() + " ");
                stringArray[i+1]=String.valueOf(sortedTopics[i].getWeight()) ;
				/*pw.print (
						  sortedTopics[i].getWeight() + " ");*/
            }
            //pw.print (" \n");
            bwriter.writeNext(stringArray);
            //pw.print(data.get(di).instance.getData() +" \n");

            Arrays.fill(topicCounts, 0);
        }
        bwriter.close();

    }

    protected void inferTopicsForOneDocOld (FeatureSequence tokenSequence,
                                            FeatureSequence topicSequence,
                                            boolean shouldSaveState,
                                            boolean readjustTopicsAndStats /* currently ignored */) {
        int[] oneDocTopics = topicSequence.getFeatures();

        TIntIntHashMap currentTypeTopicCounts;
        int type, oldTopic, newTopic;
        int docLength = tokenSequence.getLength();

        //		populate topic counts
        TIntIntHashMap localTopicCounts = new TIntIntHashMap();
        for (int position = 0; position < docLength; position++) {
            localTopicCounts.adjustOrPutValue(oneDocTopics[position], 1, 1);
        }
        /**
         // major change DP 08/20/2014
         for (int k=0;k<200;numTopics++){
         cachedCoefficients[k] =	0;
         }*/
        //		Initialize the topic count/beta sampling bucket
        double topicBetaMass = 0.0; //eqn 8
        for (int topic: localTopicCounts.keys()) {
            int n = localTopicCounts.get(topic);

            //			initialize the normalization constant for the (B * n_{t|d}) term
            topicBetaMass += beta * n /	(tokensPerTopic[topic] + betaSum);

            //			update the coefficients for the non-zero topics
            cachedCoefficients[topic] =	(alpha[topic] + n) / (tokensPerTopic[topic] + betaSum); // for each topic eqn 9
        }

        double topicTermMass = 0.0;

        double[] topicTermScores = new double[numTopics];
        int[] topicTermIndices;
        int[] topicTermValues;
        int i;
        double score;

        //	Iterate over the positions (words) in the document
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            //if (type < numTypes) {
            //System.out.println(type);
            oldTopic = oneDocTopics[position];

            currentTypeTopicCounts = typeTopicCounts[type];
            //System.out.println("typeTopicCounts" + typeTopicCounts[type].get(oldTopic) );
            //currentTypeTopicCounts.adjustValue(oldTopic, -1);
            //System.out.println("CurrenttypeTopicCounts" + currentTypeTopicCounts.get(oldTopic) );
            //System.out.println("typeTopicCounts" + typeTopicCounts[type].get(oldTopic) );*/
            assert(currentTypeTopicCounts.get(oldTopic) >= 0);

            //	Remove this token from all counts.
            //   Note that we actually want to remove the key if it goes
            //    to zero, not set it to 0.
            /**
             if (currentTypeTopicCounts.get(oldTopic) == 1) {
             currentTypeTopicCounts.remove(oldTopic);
             }
             else {
             currentTypeTopicCounts.adjustValue(oldTopic, -1);
             }*/
            /**
             smoothingOnlyMass -= alpha[oldTopic] * beta /
             (tokensPerTopic[oldTopic] + betaSum);
             */
            topicBetaMass -= beta * localTopicCounts.get(oldTopic) /
                    (tokensPerTopic[oldTopic] + betaSum);

            if (localTopicCounts.get(oldTopic) == 1) {
                localTopicCounts.remove(oldTopic);
            }
            else {
                localTopicCounts.adjustValue(oldTopic, -1);
            }

            //tokensPerTopic[oldTopic]--;
            //in our model nothing happens to the tokenspertopic but it does happen to localtopicscounts
            // so no change to smoothing only mass but change to topicBetaMass
            /**
             smoothingOnlyMass += alpha[oldTopic] * beta /
             (tokensPerTopic[oldTopic] + betaSum);
             */
            topicBetaMass += beta * localTopicCounts.get(oldTopic) /
                    (tokensPerTopic[oldTopic] + betaSum);

            cachedCoefficients[oldTopic] =
                    (alpha[oldTopic] + localTopicCounts.get(oldTopic)) /
                            (tokensPerTopic[oldTopic] + betaSum); // this varies because the local topic counts vary. Note this is is just the first half of equation 9

            topicTermMass = 0.0;

            topicTermIndices = currentTypeTopicCounts.keys();
            topicTermValues = currentTypeTopicCounts.getValues();

            for (i=0; i < topicTermIndices.length; i++) {
                int topic = topicTermIndices[i];
                score =
                        cachedCoefficients[topic] * topicTermValues[i];
                //				((alpha[topic] + localTopicCounts.get(topic)) *
                //				topicTermValues[i]) /
                //				(tokensPerTopic[topic] + betaSum);

                //				Note: I tried only doing this next bit if
                //				score > 0, but it didn't make any difference,
                //				at least in the first few iterations.

                topicTermMass += score; // equal to eqn 9
                topicTermScores[i] = score;
                //				topicTermIndices[i] = topic;
            }
            //			indicate that this is the last topic
            //			topicTermIndices[i] = -1;

            double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
            double origSample = sample;

            //			Make sure it actually gets set
            newTopic = -1;

            if (sample < topicTermMass) {
                //topicTermCount++;

                i = -1;
                while (sample > 0) {
                    i++;
                    sample -= topicTermScores[i];
                }
                newTopic = topicTermIndices[i];

            }
            else {
                sample -= topicTermMass;

                if (sample < topicBetaMass) {
                    //betaTopicCount++;

                    sample /= beta;

                    topicTermIndices = localTopicCounts.keys();
                    topicTermValues = localTopicCounts.getValues();

                    for (i=0; i < topicTermIndices.length; i++) {
                        newTopic = topicTermIndices[i];

                        sample -= topicTermValues[i] /
                                (tokensPerTopic[newTopic] + betaSum);

                        if (sample <= 0.0) {
                            break;
                        }
                    }

                }
                else {
                    //smoothingOnlyCount++;

                    sample -= topicBetaMass;

                    sample /= beta;

                    for (int topic = 0; topic < numTopics; topic++) {
                        sample -= alpha[topic] /
                                (tokensPerTopic[topic] + betaSum);

                        if (sample <= 0.0) {
                            newTopic = topic;
                            break;
                        }
                    }

                }

            }

            if (newTopic == -1) {
                System.err.println("LDAHyper sampling error: "+ origSample + " " + sample + " " + smoothingOnlyMass + " " +
                        topicBetaMass + " " + topicTermMass);
                newTopic = numTopics-1; // TODO is this appropriate
                //throw new IllegalStateException ("LDAHyper: New topic not sampled.");
            }
            //assert(newTopic != -1);

            //			Put that new topic into the counts
            oneDocTopics[position] = newTopic;
            //currentTypeTopicCounts.adjustOrPutValue(newTopic, 1, 1);
            /**
             smoothingOnlyMass -= alpha[newTopic] * beta /
             (tokensPerTopic[newTopic] + betaSum);
             */
            topicBetaMass -= beta * localTopicCounts.get(newTopic) /
                    (tokensPerTopic[newTopic] + betaSum);

            localTopicCounts.adjustOrPutValue(newTopic, 1, 1);
            //tokensPerTopic[newTopic]++;

            //			update the coefficients for the non-zero topics
            cachedCoefficients[newTopic] =
                    (alpha[newTopic] + localTopicCounts.get(newTopic)) /
                            (tokensPerTopic[newTopic] + betaSum);
            /**
             smoothingOnlyMass += alpha[newTopic] * beta /
             (tokensPerTopic[newTopic] + betaSum);
             */
            topicBetaMass += beta * localTopicCounts.get(newTopic) /
                    (tokensPerTopic[newTopic] + betaSum);

            //assert(currentTypeTopicCounts.get(newTopic) >= 0);
            //System.out.println("typeTopicCounts" + typeTopicCounts[type].get(oldTopic) );
            //} //end type check
        }

        //		Clean up our mess: reset the coefficients to values with only
        //		smoothing. The next doc will update its own non-zero topics... doesnt really clean anything
        for (int topic: localTopicCounts.keys()) {
            cachedCoefficients[topic] =
                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

        if (shouldSaveState) {
            //			Update the document-topic count histogram,
            //			for dirichlet estimation
            docLengthCounts[ docLength ]++;
            for (int topic: localTopicCounts.keys()) {
                topicDocCounts[topic][ localTopicCounts.get(topic) ]++;
            }
        }
    }


    protected void inferTopicsForOneDoc (FeatureSequence tokenSequence,
                                         FeatureSequence topicSequence,
                                         boolean shouldSaveState,
                                         boolean readjustTopicsAndStats /* currently ignored */) {
        int[] oneDocTopics = topicSequence.getFeatures();

        TIntIntHashMap currentTypeTopicCounts;
        int type, oldTopic, newTopic;
        //double topicWeightsSum;
        int docLength = tokenSequence.getLength();

        //		populate topic counts
        TIntIntHashMap localTopicCounts = new TIntIntHashMap();
        for (int position = 0; position < docLength; position++) {
            localTopicCounts.adjustOrPutValue(oneDocTopics[position], 1, 1);
        }

        //		Initialize the topic count/beta sampling bucket
        double topicBetaMass = 0.0;
        for (int topic: localTopicCounts.keys()) {
            int n = localTopicCounts.get(topic);

            //			initialize the normalization constant for the (B * n_{t|d}) term
            topicBetaMass += beta * n /	(tokensPerTopic[topic] + betaSum);

            //			update the coefficients for the non-zero topics
            cachedCoefficients[topic] =	(alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
        }

        double topicTermMass = 0.0;

        double[] topicTermScores = new double[numTopics];
        int[] topicTermIndices;
        int[] topicTermValues;
        int i;
        double score;

        //	Iterate over the positions (words) in the document
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);

            oldTopic = oneDocTopics[position];

            currentTypeTopicCounts = typeTopicCounts[type];
            assert(currentTypeTopicCounts.get(oldTopic) >= 0);

            //	Remove this token from all counts.
            //   Note that we actually want to remove the key if it goes
            //    to zero, not set it to 0.
            /**
             if (currentTypeTopicCounts.get(oldTopic) == 1) {
             currentTypeTopicCounts.remove(oldTopic);
             }
             else {
             currentTypeTopicCounts.adjustValue(oldTopic, -1);
             }
             */
            /**smoothingOnlyMass -= alpha[oldTopic] * beta /
             (tokensPerTopic[oldTopic] + betaSum);
             */
            topicBetaMass -= beta * localTopicCounts.get(oldTopic) /
                    (tokensPerTopic[oldTopic] + betaSum);

            if (localTopicCounts.get(oldTopic) == 1) {
                localTopicCounts.remove(oldTopic);
            }
            else {
                localTopicCounts.adjustValue(oldTopic, -1);
            }
            /**
             tokensPerTopic[oldTopic]--;

             smoothingOnlyMass += alpha[oldTopic] * beta /
             (tokensPerTopic[oldTopic] + betaSum);
             */
            topicBetaMass += beta * localTopicCounts.get(oldTopic) /
                    (tokensPerTopic[oldTopic] + betaSum);

            cachedCoefficients[oldTopic] =
                    (alpha[oldTopic] + localTopicCounts.get(oldTopic)) /
                            (tokensPerTopic[oldTopic] + betaSum);

            topicTermMass = 0.0;

            topicTermIndices = currentTypeTopicCounts.keys();
            topicTermValues = currentTypeTopicCounts.getValues();

            for (i=0; i < topicTermIndices.length; i++) {
                int topic = topicTermIndices[i];
                score =
                        cachedCoefficients[topic] * topicTermValues[i];
                //				((alpha[topic] + localTopicCounts.get(topic)) *
                //				topicTermValues[i]) /
                //				(tokensPerTopic[topic] + betaSum);

                //				Note: I tried only doing this next bit if
                //				score > 0, but it didn't make any difference,
                //				at least in the first few iterations.

                topicTermMass += score;
                topicTermScores[i] = score;
                //				topicTermIndices[i] = topic;
            }
            //			indicate that this is the last topic
            //			topicTermIndices[i] = -1;

            double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
            double origSample = sample;

//			Make sure it actually gets set
            newTopic = -1;

            if (sample < topicTermMass) {
                //topicTermCount++;

                i = -1;
                while (sample > 0) {
                    i++;
                    sample -= topicTermScores[i];
                }
                newTopic = topicTermIndices[i];

            }
            else {
                sample -= topicTermMass;

                if (sample < topicBetaMass) {
                    //betaTopicCount++;

                    sample /= beta;

                    topicTermIndices = localTopicCounts.keys();
                    topicTermValues = localTopicCounts.getValues();

                    for (i=0; i < topicTermIndices.length; i++) {
                        newTopic = topicTermIndices[i];

                        sample -= topicTermValues[i] /
                                (tokensPerTopic[newTopic] + betaSum);

                        if (sample <= 0.0) {
                            break;
                        }
                    }

                }
                else {
                    //smoothingOnlyCount++;

                    sample -= topicBetaMass;

                    sample /= beta;

                    for (int topic = 0; topic < numTopics; topic++) {
                        sample -= alpha[topic] /
                                (tokensPerTopic[topic] + betaSum);

                        if (sample <= 0.0) {
                            newTopic = topic;
                            break;
                        }
                    }

                }

            }

            if (newTopic == -1) {
                System.err.println("LDAHyper sampling error: "+ origSample + " " + sample + " " + smoothingOnlyMass + " " +
                        topicBetaMass + " " + topicTermMass);
                newTopic = numTopics-1; // TODO is this appropriate
                //throw new IllegalStateException ("LDAHyper: New topic not sampled.");
            }
            //assert(newTopic != -1);

            //			Put that new topic into the counts
            oneDocTopics[position] = newTopic;
            /**
             currentTypeTopicCounts.adjustOrPutValue(newTopic, 1, 1);


             smoothingOnlyMass -= alpha[newTopic] * beta /
             (tokensPerTopic[newTopic] + betaSum);
             */
            topicBetaMass -= beta * localTopicCounts.get(newTopic) /
                    (tokensPerTopic[newTopic] + betaSum);

            localTopicCounts.adjustOrPutValue(newTopic, 1, 1);
            /**
             tokensPerTopic[newTopic]++;
             */
            //			update the coefficients for the non-zero topics
            cachedCoefficients[newTopic] =
                    (alpha[newTopic] + localTopicCounts.get(newTopic)) /
                            (tokensPerTopic[newTopic] + betaSum);
            /**
             smoothingOnlyMass += alpha[newTopic] * beta /
             (tokensPerTopic[newTopic] + betaSum);
             */
            topicBetaMass += beta * localTopicCounts.get(newTopic) /
                    (tokensPerTopic[newTopic] + betaSum);

            assert(currentTypeTopicCounts.get(newTopic) >= 0);

        }

        //		Clean up our mess: reset the coefficients to values with only
        //		smoothing. The next doc will update its own non-zero topics...
        for (int topic: localTopicCounts.keys()) {
            cachedCoefficients[topic] =
                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

        if (shouldSaveState) {
            //			Update the document-topic count histogram,
            //			for dirichlet estimation
            docLengthCounts[ docLength ]++;
            for (int topic: localTopicCounts.keys()) {
                topicDocCounts[topic][ localTopicCounts.get(topic) ]++;
            }
        }
    }

    public void optimizeBeta() {
        // The histogram starts at count 0, so if all of the
        //  tokens of the most frequent type were assigned to one topic,
        //  we would need to store a maxTypeCount + 1 count.
        // so if "hello" occurs 500 times, you ned a histogram from 0 to 500 (included)
        // so you need 501
        int[] countHistogram = new int[maxTypeCount + 1];

        // Now count the number of type/topic pairs that have
        //  each number of tokens.

        int index;
        for (int type = 0; type < numTypes; type++) {
            int[] counts = typeTopicCounts[type].getValues();


            index = 0;

            while (index < counts.length &&
                    counts[index] > 0) {
                /**so go through all the indices of the type (i.e. all topics)
                 get the counts from each index.
                 so this type and topic pair have a count =  5 and for counthistogram[5]
                 we increment the value  by 1.

                 **/
                int count = counts[index] ;
                //System.out.println("type: " + type +"length: "+counts.length+ "topic: " + index + "count: " +count);
                countHistogram[count]++;
                index++;
            }

        }

        // Figure out how large we need to make the "observation lengths"
        //  histogram.
        int maxTopicSize = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            if (tokensPerTopic[topic] > maxTopicSize) {
                maxTopicSize = tokensPerTopic[topic];
            }
        }

        // Now allocate it and populate it.
        int[] topicSizeHistogram = new int[maxTopicSize + 1];
        for (int topic = 0; topic < numTopics; topic++) {
            topicSizeHistogram[ tokensPerTopic[topic] ]++;
        }

        betaSum = Dirichlet.learnSymmetricConcentration(countHistogram,
                topicSizeHistogram,
                numTypes,
                betaSum);
        beta = betaSum / numTypes;

        System.out.println("new Beta: " +beta);

    }

    private void initializeForTypes(Alphabet alphabet) {
        if (this.alphabet == null) {
            this.alphabet = alphabet;
            this.numTypes = alphabet.size();
            this.typeTopicCounts = new TIntIntHashMap[this.numTypes];

            for(int fi = 0; fi < this.numTypes; ++fi) {
                this.typeTopicCounts[fi] = new TIntIntHashMap();
            }

            this.betaSum = this.beta * (double)this.numTypes;
        } else {
            if (alphabet != this.alphabet) {
                throw new IllegalArgumentException("Cannot change Alphabet.");
            }

            if (alphabet.size() != this.numTypes) {
                this.numTypes = alphabet.size();
                TIntIntHashMap[] newTypeTopicCounts = new TIntIntHashMap[this.numTypes];

                int i;
                for(i = 0; i < this.typeTopicCounts.length; ++i) {
                    newTypeTopicCounts[i] = this.typeTopicCounts[i];
                }

                for(i = this.typeTopicCounts.length; i < this.numTypes; ++i) {
                    newTypeTopicCounts[i] = new TIntIntHashMap();
                }

                this.betaSum = this.beta * (double)this.numTypes;
            }
        }

    }


}
