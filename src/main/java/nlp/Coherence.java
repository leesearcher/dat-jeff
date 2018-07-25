package cc.mallet.topics;




import gnu.trove.TIntHashSet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;



import java.util.Collections;


import cc.mallet.types.FeatureSequence;
import au.com.bytecode.opencsv.CSVWriter;
//import gnu.trove.set.hash.TIntHashSet;


public class Coherence {
	
	class kldistances{
		
		private double avg_klfrom_uniform;
		private double [] klfrom_uniform;
		
		public kldistances(double avg_klfrom_uniform , double[] klfrom_uniform ){
			this.avg_klfrom_uniform =avg_klfrom_uniform;
			this.klfrom_uniform =klfrom_uniform;
			
		}
		
		public double getAvgKlfromUniform(){
			return avg_klfrom_uniform;	
		}
		public double[] getKlfromUniform(){
			return klfrom_uniform;	
		}
		
	}
	
	int[][][] topicCodocumentMatrices;
	
	@SuppressWarnings("deprecation")
	public void collectDocumentStatistics (int numTopics, int numTopWords, int V, gnu.trove.TIntIntHashMap[] typeTopicCounts ,DMRTopicModelXBeta model, String dir, String modelName ){
		
		
		//DMRTopicModelXBeta model = new DMRTopicModelXBeta(4);
		 
		
		int numTypes = typeTopicCounts.length;
		double[][] phi = new double[numTopics][numTypes]; //K x V

		
		for (int t=0;t< numTypes;t++){
			for (int topic=0; topic<numTopics; topic++){
				phi[topic][t] = 0;
				if (typeTopicCounts[t].containsKey(topic)){
					phi[topic][t] = typeTopicCounts[t].get(topic);
				} // end if
			}//end topic
		} // end types
		
		for (int topic=0; topic<numTopics; topic++){
			double total = 0;
			for (int t1=0;t1< numTypes;t1++){
				total = total+ phi[topic][t1];
				}
			for (int t1=0;t1< numTypes;t1++){
				phi[topic][t1] = phi[topic][t1]/total ;
				}
		}
		kldistances KLdistances = KLfromUniform ( numTopics,  V,  phi );
		double avg_klfrom_uniform= KLdistances.getAvgKlfromUniform();
		double[] distFromUniform = KLdistances.getKlfromUniform();
		TIntHashSet[] docWordSets = new TIntHashSet[model.data.size()]; //model.data.size();
		
		
		
		topicCodocumentMatrices = new int[numTopics][numTopWords][numTopWords];
		
		int[][] topicTopWords = new int[numTopics][numTopWords];
		
		TIntHashSet[] topicTopWordSet = new TIntHashSet[numTopics];
		
		double[] Coherence = new double[numTopics];
		double AvgTopicCoherence=0;
		//Sort the topics
		//now capture the top words of a topic.
		for (int topic=0;topic<numTopics;topic++){
		    ArrayList<Pair> wordsProbsList = new ArrayList<Pair>(); 
		    TIntHashSet givenTopic = new TIntHashSet();
	        for (int w = 0; w < V; w++){
	            Pair p = new Pair(w, phi[topic][w], false);
	            wordsProbsList.add(p);
	        }//end for each word
	        Collections.sort(wordsProbsList);
	        // Collect the top words in a hash set and in a matrix
	        for (int i=0;i<numTopWords;i++){
	        	topicTopWords[topic][i]=(Integer) wordsProbsList.get(i).first;
	        	int x = topicTopWords[topic][i];
	        	//System.out.println("Coherence x "+x+topic);
		        givenTopic.add(x);
		    }//go through top words
	        topicTopWordSet[topic]=givenTopic;
        }//close topic loop
		
		
		
		//Created a HashSet for documents - may not be a good idea!!But I am restricting the number of words in the document has set.
		for (int m_ = 0; m_ < model.data.size(); m_++){       	//model.data.size();
			
			FeatureSequence fs = (FeatureSequence) model.data.get(m_).instance.getData();
			int N = fs.getLength();
			
        	//int N = data.docs.get(m_).length;       //model.data.;
        	TIntHashSet givenDoc = new TIntHashSet();
        	for (int n = 0; n < N; n++){       		
        		//Fix
                int w = fs.getIndexAtPosition(n);// data.docs.get(m_).words[n];
                
                for (int topic=0;topic<numTopics;topic++){
                	if(topicTopWordSet[topic].contains(w)){
                		givenDoc.add(w);
                	} // close if condition
                }//close inner topic loop
            docWordSets[m_] = givenDoc;
        	}//close loop for one document
		}//close loop for all documents
		String headers="Topic Number, Topic Name , Coherence, DIstFromUniform, TitleCodocFrequency "; 
		String[][] cohereResults = new String[numTopics+1][5];
		cohereResults[0]= headers.split(",");
		System.out.println(headers);
		
		for (int topic = 0; topic < numTopics; topic++){
			//Topic Loop
			int[] indices = topicTopWords[topic];
			Coherence[topic]=0;
            for (int m = 0; m < model.data.size(); m++){
            	//Document Loop Within Topic
            	TIntHashSet supportedWords = docWordSets[m];
	            for (int i = 0; i < numTopWords; i++) {
					if (supportedWords.contains(indices[i])) {
						for (int j = i; j < numTopWords; j++) {
							if (i == j) {
							// Diagonals are total number of documents with word W in topic T
								topicCodocumentMatrices[topic][i][i]++;
							}
							else if (supportedWords.contains(indices[j])) {
								topicCodocumentMatrices[topic][i][j]++;
								topicCodocumentMatrices[topic][j][i]++;
							}//end else if
						}//end inner loop
					}//end if condition
				} //end for loop over top words
	         } //end loop over documents
         
            //Coherence for kth topic
            
            //int MaxWordPair_1=0;
            //int MaxWordPair_2=0;
            //double maxCohere=-5000;
            for (int i = 1; i < numTopWords; i++) {
            	//for (int j = 0; j < i; j++) {
				for (int j = 0; j < i-1; j++) {
					double denom = Math.log(topicCodocumentMatrices[topic][j][j]);
					double instCohere=Math.log(topicCodocumentMatrices[topic][i][j] +1)- denom;
					Coherence[topic] = Coherence[topic]+instCohere  ;
					/**if (instCohere > maxCohere ){
						MaxWordPair_1=i;
						MaxWordPair_2=j;
						maxCohere=instCohere;
					}*/
				}// end inner loop
            }// end outer loop over top topic words to get to coherence
            
            //int MaxWordPair_1=indices[0];
            //int MaxWordPair_2=indices[1];
            String FirstWord = (String) model.alphabet.lookupObject(indices[0]) ; //data.localDict.getWord(indices[MaxWordPair_1]);
            String SecondWord = (String) model.alphabet.lookupObject(indices[1]); //data.localDict.getWord(indices[MaxWordPair_2]);
            
            //System.out.println(topic+","+ FirstWord+":"+SecondWord + "," +Coherence[topic]+ "," +  distFromUniform[topic] +","+ topicCodocumentMatrices[topic][MaxWordPair_1][MaxWordPair_2] );
            System.out.println(topic+","+ FirstWord+":"+SecondWord + "," +Coherence[topic]+ "," +  distFromUniform[topic] );
            cohereResults[topic+1]=( String.valueOf(topic)+","+ FirstWord+":"+SecondWord + "," +String.valueOf(Coherence[topic])+ "," +  String.valueOf(distFromUniform[topic])  ).split(",");// +","+ String.valueOf(topicCodocumentMatrices[topic][MaxWordPair_1][MaxWordPair_2])  ).split(",");
            //cohereResults[topic+1]=(String.valueOf(topic)+","+ FirstWord+":"+SecondWord + "," +String.valueOf(Coherence[topic])+ "," +  String.valueOf(distFromUniform[topic]) ;
            AvgTopicCoherence +=Coherence[topic] ;
            
	}// end loop over topics
	try {
		CSVWriter csvwriter  = new CSVWriter(new FileWriter(dir+File.separator+modelName+"CoherenceTopics.csv"), ',');
		for(int num=0;num<=numTopics;num++){
			csvwriter.writeNext(cohereResults[num]);}
		csvwriter.close();
	} 
	catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
		}
	
	
	double[] divergences= 	DivergenceScores (numTopics,  V, phi );
	AvgTopicCoherence=AvgTopicCoherence/numTopics;
	System.out.println("AvgTopicCoherence," +"AvgTopicKL From Uniform," +"JSDIvergence," +"COSDIvergence");
	System.out.println(AvgTopicCoherence +"," +avg_klfrom_uniform+"," +divergences[0]+"," +divergences[1] +"\n");
	}//end collectDocumentStatistics
	
	//KL Divergence From Uniform symmetric:
	public kldistances KLfromUniform (int numTopics, int V, double[][] phi ){
		//need proportions!
		double [] kldistanceArray = new double[numTopics];
		double AvgKLFromUniform=0;
		double [] topicSum = new double[numTopics];
		//double [] topicTokens = new double[numTopics];
		for (int topic=0; topic<numTopics;topic++){
			topicSum[topic] =0;
			//topicTokens[topic]=0;
			for(int word=0;word<V;word++){
				topicSum[topic] += phi[topic][word];
				if (phi[topic][word]>0){
					//topicTokens[topic]+=1;
				}// end if condition
			}//loop over vocabulary
			//System.out.println("KL DistFromUniform topic :"+topicSum );
		}//end Topic Loop
		
		
		for (int topic=0; topic<numTopics;topic++){
			double distFromUniform=0;
			for(int word=0;word<V;word++){
				distFromUniform += 0.5*(((phi[topic][word]*1.0)/topicSum[topic]) -(1.0/V)) * Math.log((phi[topic][word]+1e-12 ) * ((V*1.0)/ topicSum[topic])) ;
			}//loop over vocabulary
			//System.out.println("\n KL DistFromUniform topic:   " +topic+"  :"+ distFromUniform + "  "+ topicTokens[topic] );
			AvgKLFromUniform += distFromUniform ;
			kldistanceArray[topic]= distFromUniform;
		}//end Topic Loop
		
		AvgKLFromUniform=AvgKLFromUniform/numTopics;
		//System.out.println("\t,AvgKLFromUniform:"+AvgKLFromUniform);
		
		kldistances result = new kldistances(AvgKLFromUniform,kldistanceArray );
		return result;
	}//end KLfromUniform

	public double[] DivergenceScores (int numTopics, int V, double[][] phi ){
		// this is to compute the average Jensen Shannon divergence amongst topics from each model. This is a nice symmetric measure. 
		double [] topicSum = new double[numTopics];
		double [] topicNorm = new double[numTopics];
		double jsDivergence=0;
		double cosDivergence =0;
		
		for (int topic=0; topic<numTopics;topic++){
			topicSum[topic] =0;
			topicNorm[topic] =0;
			for(int word=0;word<V;word++){
				topicSum[topic] += phi[topic][word];
				topicNorm[topic] += Math.pow(phi[topic][word],2);
			}//loop over vocabulary	
			topicNorm[topic] = Math.sqrt(topicNorm[topic]);
		}//end Topic Loop
		
		int numComparisons =0;
		for (int topic=0; topic<numTopics;topic++){
			for (int nextTopic=topic+1; nextTopic<numTopics;nextTopic++){
				
				double js_mean = 0;
				double cosDivergenceInst=0;
				double jsDivergenceInst=0;
			
				for(int word=0;word<V;word++){
					double curTop=(phi[topic][word]/topicSum[topic]);
					double nexTop = (phi[nextTopic][word]/topicSum[nextTopic]);
					
					js_mean = 0.5*( curTop + nexTop); // simply get to Mean  for Jensen Shannon Divergence
					if(js_mean>0){
					if(curTop>0){
						jsDivergenceInst += 0.5*(curTop) * Math.log( curTop/ js_mean );
						}//curTop condn
					if(nexTop>0){
						jsDivergenceInst += 0.5*(nexTop) * Math.log( nexTop/ js_mean );
						}//nexTop condn
								
					}//js_mean COndn
					//System.out.println("check,:   "+":"+topic +":"+js_mean +":"+jsDivergenceInst+":"+curTop+":"+nexTop );
					cosDivergenceInst += (phi[topic][word]*phi[nextTopic][word]);
					
				}//loop over vocabulary
				numComparisons +=1;
				//System.out.println("\t, COSDivergence Inst:   " +cosDivergenceInst +":"+jsDivergenceInst );
				
				cosDivergence +=cosDivergenceInst/(topicNorm[topic] * topicNorm[nextTopic]);
				
				jsDivergence +=Math.sqrt(jsDivergenceInst);
			}// next Topic Loop
			
		}//end Topic Loop
		jsDivergence=jsDivergence/numComparisons;
		cosDivergence=cosDivergence/numComparisons;
		double[] results= new double[2];
		results[0]=jsDivergence;
		results[1]=cosDivergence ;
		return results;
	}//end class DivergenceScores
	
	
}//end coherence
		/**
		kldistances KLdistances = KLfromUniform ( numTopics,  V,  phi );
		double avg_klfrom_uniform= KLdistances.getAvgKlfromUniform();
		double[] distFromUniform = KLdistances.getKlfromUniform();
		TIntHashSet[] docWordSets = new TIntHashSet[model.data.size()]; //model.data.size();
		
		
		
		topicCodocumentMatrices = new int[numTopics][numTopWords][numTopWords];
		
		int[][] topicTopWords = new int[numTopics][numTopWords];
		
		TIntHashSet[] topicTopWordSet = new TIntHashSet[numTopics];
		
		double[] Coherence = new double[numTopics];
		double AvgTopicCoherence=0;
		//Sort the topics
		//now capture the top words of a topic.
		for (int topic=0;topic<numTopics;topic++){
		    ArrayList<Pair> wordsProbsList = new ArrayList<Pair>(); 
		    TIntHashSet givenTopic = new TIntHashSet();
	        for (int w = 0; w < V; w++){
	            Pair p = new Pair(w, phi[topic][w], false);
	            wordsProbsList.add(p);
	        }//end for each word
	        Collections.sort(wordsProbsList);
	        // Collect the top words in a hash set and in a matrix
	        for (int i=0;i<numTopWords;i++){
	        	topicTopWords[topic][i]=(Integer) wordsProbsList.get(i).first;
	        	int x = topicTopWords[topic][i];
	        	//System.out.println("Coherence x "+x+topic);
		        givenTopic.add(x);
		    }//go through top words
	        topicTopWordSet[topic]=givenTopic;
        }//close topic loop
		
		
		
		//Created a HashSet for documents - may not be a good idea!!But I am restricting the number of words in the document has set.
		for (int m_ = 0; m_ < model.data.size(); m_++){       	//model.data.size();
			
			FeatureSequence fs = (FeatureSequence) model.data.get(m_).instance.getData();
			int N = fs.getLength();
			
        	//int N = data.docs.get(m_).length;       //model.data.;
        	TIntHashSet givenDoc = new TIntHashSet();
        	for (int n = 0; n < N; n++){       		
        		//Fix
                int w = fs.getIndexAtPosition(n);// data.docs.get(m_).words[n];
                
                for (int topic=0;topic<numTopics;topic++){
                	if(topicTopWordSet[topic].contains(w)){
                		givenDoc.add(w);
                	} // close if condition
                }//close inner topic loop
            docWordSets[m_] = givenDoc;
        	}//close loop for one document
		}//close loop for all documents
		String headers="Topic Number, Topic Name , Coherence, DIstFromUniform, TitleCodocFrequency "; 
		String[][] cohereResults = new String[numTopics+1][5];
		cohereResults[0]= headers.split(",");
		System.out.println(headers);
		
		for (int topic = 0; topic < numTopics; topic++){
			//Topic Loop
			int[] indices = topicTopWords[topic];
			Coherence[topic]=0;
            for (int m = 0; m < model.data.size(); m++){
            	//Document Loop Within Topic
            	TIntHashSet supportedWords = docWordSets[m];
	            for (int i = 0; i < numTopWords; i++) {
					if (supportedWords.contains(indices[i])) {
						for (int j = i; j < numTopWords; j++) {
							if (i == j) {
							// Diagonals are total number of documents with word W in topic T
								topicCodocumentMatrices[topic][i][i]++;
							}
							else if (supportedWords.contains(indices[j])) {
								topicCodocumentMatrices[topic][i][j]++;
								topicCodocumentMatrices[topic][j][i]++;
							}//end else if
						}//end inner loop
					}//end if condition
				} //end for loop over top words
	         } //end loop over documents
         
            //Coherence for kth topic
            int MaxWordPair_1=0;
            int MaxWordPair_2=0;
            double maxCohere=-5000;
            for (int i = 1; i < numTopWords; i++) {
            	//for (int j = 0; j < i; j++) {
				for (int j = 0; j < i-1; j++) {
					double denom = Math.log(topicCodocumentMatrices[topic][j][j]);
					double instCohere=Math.log(topicCodocumentMatrices[topic][i][j] +1)- denom;
					Coherence[topic] = Coherence[topic]+instCohere  ;
					if (instCohere > maxCohere ){
						MaxWordPair_1=i;
						MaxWordPair_2=j;
						maxCohere=instCohere;
					}
				}// end inner loop
            }// end outer loop over top topic words to get to coherence
            String FirstWord = (String) model.alphabet.lookupObject(indices[MaxWordPair_1]) ; //data.localDict.getWord(indices[MaxWordPair_1]);
            String SecondWord = (String) model.alphabet.lookupObject(indices[MaxWordPair_2]); //data.localDict.getWord(indices[MaxWordPair_2]);
            
            System.out.println(topic+","+ FirstWord+":"+SecondWord + "," +Coherence[topic]+ "," +  distFromUniform[topic] +","+ topicCodocumentMatrices[topic][MaxWordPair_1][MaxWordPair_2] );
            cohereResults[topic+1]=(String.valueOf(topic)+","+ FirstWord+":"+SecondWord + "," +String.valueOf(Coherence[topic])+ "," +  String.valueOf(distFromUniform[topic]) +","+ String.valueOf(topicCodocumentMatrices[topic][MaxWordPair_1][MaxWordPair_2])  ).split(",");
            AvgTopicCoherence +=Coherence[topic] ;
            
	}// end loop over topics
	try {
		CSVWriter csvwriter  = new CSVWriter(new FileWriter(dir+File.separator+modelName+"CoherenceTopics.csv"), ',');
		for(int num=0;num<=numTopics;num++){
			csvwriter.writeNext(cohereResults[num]);}
		csvwriter.close();
	} 
	catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
		}
	
	
	//double[] divergences= 	DivergenceScores (numTopics,  V, phi );
	//AvgTopicCoherence=AvgTopicCoherence/numTopics;
	//System.out.println("AvgTopicCoherence," +"AvgTopicKL From Uniform," +"JSDIvergence," +"COSDIvergence");
	//System.out.println(AvgTopicCoherence +"," +avg_klfrom_uniform+"," +divergences[0]+"," +divergences[1] +"\n");
		
	
	
	}//end collectDocumentStatistics
	*/
