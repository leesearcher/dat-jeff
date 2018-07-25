package nlp;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import cc.mallet.topics.tui.DMRLoader;
import cc.mallet.topics.tui.Perplexity;
import cc.mallet.types.InstanceList;

public class CaloriesLda {
@SuppressWarnings("deprecation")
public static void main (String[] args) throws IOException {

	int samples = 500;	
	int pseudoCount = 5;
	int randomSeed = 10;
	int burnin = 10000;
	int optimInterval = 100;
	int numIters = 50000 ;
	int inferIters = 3000;
	String subDir="FINALSIMPLE/";
	// nocolon/, short/
	
	System.out.println(
			"type = " + subDir + '\n' +
			"pseudoCount = " + pseudoCount + '\n' + 
			"randomSeed = " + randomSeed + '\n' +
			"burnin = " + burnin + '\n' +
			"optimInterval = " + optimInterval + '\n' +
			"numIters = " + numIters + '\n' +
			"inferIters = " + inferIters + '\n'
			);
	
	
	String processed_data="D:/Dinesh/Projects/CALORIES/MKTSC/Data/";

	
	
	
	
	String subDir_out = subDir + "FINALSIMPLE";
	
	if (pseudoCount>0){
		subDir_out = subDir + "FINALSIMPLE";
	}

	
	String apath ="D:/Dinesh/Projects/CALORIES/MKTSC/Output/Chain/"+subDir_out+Integer.toString(randomSeed)+"_x";
	new File(apath).mkdirs();
	File wordsFile = new File(processed_data+"/chain/"+subDir+"chain_text.txt");
	File featuresFile = new File(processed_data+"/chain/"+subDir+"chain_features.txt");
	File instancesFile = new File(processed_data+"/chain/"+subDir+"chain_instances.txt");

	DMRLoader loader = new DMRLoader();
	loader.load(wordsFile, featuresFile, instancesFile);
	System.out.println("Done Training Load");
	
	
	String apath_t ="D:/Dinesh/Projects/CALORIES/MKTSC/Output/NonChain/"+subDir_out+Integer.toString(randomSeed)+"_x";
	new File(apath_t).mkdirs();
	File wordsFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_text.txt");
	File featuresFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_features.txt");
	File instancesFile_t = new File(processed_data+"/non_chain/"+subDir+"nonchain_instances.txt");

	DMRLoader loader_t = new DMRLoader();
	loader_t.load(wordsFile_t, featuresFile_t, instancesFile_t);
	System.out.println("Done Testing Load");
	
	
	
	
	String[] argss = new String[3];
	argss[0] = processed_data+"/chain/"+subDir+"chain_instances.txt" ;
	
	argss[1] = "200" ;
	argss[2] = processed_data+"/non_chain/"+subDir+"nonchain_instances.txt" ;
	
	InstanceList training = InstanceList.load (new File(argss[0]));


    int numTopics = argss.length > 1 ? Integer.parseInt(argss[1]) : 20;
	

    InstanceList testing =
        argss.length > 2 ? InstanceList.load (new File(argss[2])) : null;

    DMRTopicModelXBeta lda = new DMRTopicModelXBeta (numTopics);
    lda.setRandomSeed(randomSeed);
    //lda.defaultFeatureIndex=0;
    lda.setBurninPeriod(burnin);
	lda.setOptimizeInterval(optimInterval);
	lda.setSaveStateInterval(100);
	lda.setPrintLogLikelihood(true);
	lda.setTopicDisplay(100, 10);
	lda.setPseudoCount(pseudoCount);
	lda.setSeededTopics(Arrays.asList("calories","calorie", "fat", "diet", "health", "healthy", "light", "fit", "cardio", "lean","protein" ));
	lda.typeTopicsFile = apath+"/typeTopics";
	lda.addInstances(training);
	lda.setNumIterations(numIters);
	lda.save_thrshld = 9000 ;
	//lda.save_thrshld =numIters-samples*optimInterval ;
	lda.setTopicProportionFile(apath+"/topicProportions_");
	lda.setBeta(0.01);
	//lda.dmrParameters.setParameter(0, 0, 0);
	lda.estimate();
	lda.writeParameters(new File(apath+"/dmr.parameters_"));
	lda.printState(new File(apath+"/dmr.state_.gz"));
	
	Coherence cohere = new Coherence();
	
	cohere.collectDocumentStatistics( lda.numTopics(), 20, lda.numTypes(),
			lda.getTypeTopicCounts() ,lda, apath, "train");
    //Perplexity perp = new Perplexity();
	Perplexity.calc(lda);
	//Inference
	//lda.save_thrshld = inferIters-optimInterval ;
	lda.save_thrshld = 1000 ;
	lda.iterationsSoFar = 0;
	lda.typeTopicsFile = apath_t+"/typeTopics_";
	lda.setTopicProportionFile(apath_t+"/topicProportions_");
	
	lda.addTestInstances(testing);
	lda.infer(inferIters);
	lda.writeParameters(new File(apath_t+"/dmr.parameters"));
	lda.printState(new File(apath_t+"/dmr.state.gz"));
	
		
    }

}
