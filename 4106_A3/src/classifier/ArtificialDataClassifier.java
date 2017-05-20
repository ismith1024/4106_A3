package classifier;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import classifier.KruskalTree.Graph.Edge;
import classifier.WineSamplePoint;

/*
 * 1.2.2 Training and Testing
With regard to training and testing, do the following:
1. Use a 5-fold cross-validation scheme for training and testing.
2. Using estimates of the vi;j 's, estimate the true but unknown Dependence Tree. Record
the results of how good your estimate of the true but unknown Dependence Tree is.
3. Perform a Bayesian classication1 assuming that all the random variables are independent.
4. Perform a Bayesian classication assuming that all the random variables are dependent
based on the dependence tree that you have inferred.
5. Perform the classication based on a DT algorithm. For the DT algorithm, have your
program output the resulting DT. The output2
should be neatly indented for easy viewing.
  
 * */



public class ArtificialDataClassifier {
	
	static final int FIVEFOLD_SET = 4;
	
	static Connection database;
	static Statement stat;
	
	static final int NUMSAMPLES = 2000;
	static final int NUMFEATURES = 10;
	
	static final double DT_THRESH = 0.97;
	
	//estimated and cross-validation count for the computed feature dependencies
	float[][] w1vals;
	float[][] w2vals;
	float[][] w3vals;
	float[][] w4vals;

	//confusion matrices
	float[][] confusion1;
	float[][] confusion2;
	float[][] confusion3;
	float[][] confusion4;
	
	float confusionIndependent[][];
	float confusionDependent[][];
	float confusionDecision[][];
	
	static ArrayList<ADSamplePoint> data;
	static ArrayList<ADSamplePoint> training;
	ArrayList<ADSamplePoint> testing;
	
	float[][] mim;
	
	KruskalTree tree;
	
	public ArtificialDataClassifier(){
		data = new ArrayList<>();
		testing = new ArrayList<>();
		training = new ArrayList<>();
		
		w1vals = new float[10][5];
		w2vals = new float[10][5];
		w3vals = new float[10][5];
		w4vals = new float[10][5];
		
		confusion1 = new float[10][10];
		confusion2 = new float[10][10];
		confusion3 = new float[10][10];
		confusion4 = new float[10][10];
		
		confusionIndependent = new float[4][4];	
		confusionDependent = new float[4][4];
		confusionDecision = new float[4][4];

	}
	
	public static void main(String[] args) {
		
		ArtificialDataClassifier classifier = new ArtificialDataClassifier();
		
		classifier.run();
	}
	
	public void run(){
		//get the data from the database
		getData();
		
		System.out.println("===================================\nBayesian Decision Tree:");
		//bayesian decision tree
		decisionTree();

		//independent baeysian classification
		System.out.println("Independent Bayesian Classification:");
		independentBayes();
		
		//estimate the dependence tree
		System.out.println("Learn the dependence tree topology:");
		estimateDependenceTree();
		
		System.out.println("===================================\nBayesian Dependence Tree Classification:");
		//dependent bayesian classification
		dependentBayes();
		
		

		
	}
		
	public void independentBayes() {
		//five-fold cross validation
		//will train with samples 0-1599, 2000-3599, 4000-5599, 6000-7599
		//will test with samples 1600-1999, 3600-3699, 5600-5699, 7600-7699
		float[] w1probs = new float[10];
		float[] w2probs = new float[10];
		float[] w3probs = new float[10];
		float[] w4probs = new float[10];
		
		for(float[]i: confusionIndependent) for(float j: i) j = 0;
		
		sortForTraining(FIVEFOLD_SET);
		
		int training1 = 1;
		int training2 = 1;
		int training3 = 1;
		int training4 = 1; //changed from 0 to suppress *0
		
		for(ADSamplePoint p: training){
		
			switch(p.theClass){
			case 1:
				for(int j = 0; j < NUMFEATURES; ++j) w1probs[j] += p.features[j];
				training1++;
				break;
			case 2:
				for(int j = 0; j < NUMFEATURES; ++j) w2probs[j] += p.features[j];
				training2++;
				break;
			case 3:
				for(int j = 0; j < NUMFEATURES; ++j) w3probs[j] += p.features[j];
				training3++;
				break;
			case 4:
				for(int j = 0; j < NUMFEATURES; ++j) w4probs[j] += p.features[j];
				training4++;
				break;				
			} 
		}
		
		for(int i = 0; i < NUMFEATURES; ++i){
			w1probs[i] /= training1;
			w2probs[i] /= training2;
			w3probs[i] /= training3;
			w4probs[i] /= training4;
		}
		
		//test w1:
		
		for(ADSamplePoint p: testing){ //(int i = 1600; i < 2000; ++i){
			float p1 = 1.0f;
			float p2 = 1.0f;
			float p3 = 1.0f;
			float p4 = 1.0f;
			
			for(int j = 0; j < 10; ++j){
				if(p.features[j] == 1) p1 *= w1probs[j];
				else p1 *= (1-w1probs[j]);
				if(p.features[j] == 1) p2 *= w2probs[j];
				else p2 *= (1-w2probs[j]);
				if(p.features[j] == 1) p3 *= w3probs[j];
				else p3 *= (1-w3probs[j]);
				if(p.features[j] == 1) p4 *= w2probs[j];
				else p4 *= (1-w4probs[j]);
			}
			
			//make pairwise
			if(p1 >= p2 && p1 >= p3 && p1 >= p4){
				p.classifierGuess = 1;
			} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
				p.classifierGuess = 2;
			} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
				p.classifierGuess = 3;
			} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
				p.classifierGuess = 4;
			} else p.classifierGuess = 9999;
			
			//print(data.get(i));
			confusionIndependent[p.theClass -1][p.classifierGuess -1] ++;
		}
		
////scrap comes from here
		
		System.out.println("\n..................\nConfusion matrix:");
		for(float[]i: confusionIndependent){
			System.out.print("\n| ");
			for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%2.3f", 100 *(0.0f + j)/400 ) + "% |");
			}			
		}
		
		System.out.println("\n..................");
		
	}
	
	public void dependentBayes(){
		//five-fold cross validation
		//will train with samples 0-1599, 2000-3599, 4000-5599, 6000-7599
		//will test with samples 1600-1999, 3600-3699, 5600-5699, 7600-7699
		
		//These represent w1probs0[n] = probability that value n is true given its parent is 0
		float[] w1probs0 = new float[10];
		float[] w2probs0 = new float[10];
		float[] w3probs0 = new float[10];
		float[] w4probs0 = new float[10];
		float[] w1probs1 = new float[10];
		float[] w2probs1 = new float[10];
		float[] w3probs1 = new float[10];
		float[] w4probs1 = new float[10];
		
		//counts the number of zeros in each feature
		float[] w1Count = new float[10];
		float[] w2Count = new float[10];
		float[] w3Count = new float[10];
		float[] w4Count = new float[10];
		
		for(int i = 0; i < 10; ++i){
			w1probs0[i] = 0.001f;
			w2probs0[i] = 0.001f;
			w3probs0[i] = 0.001f;
			w4probs0[i] = 0.001f;
			w1probs1[i] = 0.001f;
			w2probs1[i] = 0.001f;
			w3probs1[i] = 0.001f;
			w4probs1[i] = 0.001f;
			w1Count[i] = 0.001f;
			w2Count[i] = 0.001f;
			w3Count[i] = 0.001f;
			w4Count[i] = 0.001f;
		} //made up to init to 1
		
		for(float[]i: confusionDependent) for(float j: i) j = 0; //changed from 0
		
		sortForTraining(FIVEFOLD_SET);
		
		//train
		int rootStart = tree.rootval;
		
		int training1 = 0;
		int training2 = 0;
		int training3 = 0;
		int training4 = 0; //changed from 0
		
		for(ADSamplePoint p: training){ //(int i = 0; i < 1600; ++i){
			if(p.theClass ==1){
				w1probs0[rootStart] += p.features[rootStart];
				w1probs1[rootStart] += p.features[rootStart];
				w1Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0)
						w1probs0[e.dest]++;// += data.get(i).features[e.dest];
					else w1probs1[e.dest]++;// += data.get(i).features[e.dest];
					w1Count[e.dest]++;
				}
				
			}		
		//train w2
			else if(p.theClass == 2){ //for(int i = 2000; i < 3600; ++i){
				w2probs0[rootStart] += p.features[rootStart];
				w2probs1[rootStart] += p.features[rootStart];
				w2Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0)
						w2probs0[e.dest]++;// += data.get(i).features[e.dest];
					else w2probs1[e.dest]++;// += data.get(i).features[e.dest];
					w2Count[e.dest]++;
				}
				
			}		
			//train w3
			else if(p.theClass == 3){//(int i = 4000; i < 5600; ++i){
				w3probs0[rootStart] += p.features[rootStart];
				w3probs1[rootStart] += p.features[rootStart];
				w3Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0)
						w3probs0[e.dest]++; // += data.get(i).features[e.dest];
					else w3probs1[e.dest]++; // += data.get(i).features[e.dest];
					w3Count[e.dest]++;
				}
				
			}
			
			//train w4
			else if(p.theClass ==4){ //for(int i = 6000; i < 7600; ++i){
				w4probs0[rootStart] += p.features[rootStart];
				w4probs1[rootStart] += p.features[rootStart];
				w4Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0)
						w4probs0[e.dest]++;// += data.get(i).features[e.dest];
					else w4probs1[e.dest]++; // += data.get(i).features[e.dest];
					w4Count[e.dest]++;
				}				
			}			
		}
		
		w1Count[rootStart] = 1600;
		w2Count[rootStart] = 1600;
		w3Count[rootStart] = 1600;
		w4Count[rootStart] = 1600;
		for(int i = 0; i < NUMFEATURES; ++i){
			w1probs0[i] /= w1Count[i];
			w1probs1[i] /= w1Count[i];
			w2probs0[i] /= w2Count[i];
			w2probs1[i] /= w2Count[i];
			w3probs0[i] /= w3Count[i];
			w3probs1[i] /= w3Count[i];
			w4probs0[i] /= w4Count[i];
			w4probs1[i] /= w4Count[i];
		}
		
		w1probs0[rootStart] = 1 - w1probs0[rootStart];
		w2probs0[rootStart] = 1 - w2probs0[rootStart];
		w3probs0[rootStart] = 1 - w3probs0[rootStart];
		w4probs0[rootStart] = 1 - w4probs0[rootStart];
		
		/*for(int i = 0; i < NUMFEATURES; ++i){
			w1probs0[i] = 1 - w1probs0[i];
			w2probs0[i] = 1 - w2probs0[i];
			w3probs0[i] = 1 - w3probs0[i];
			w4probs0[i] = 1 - w4probs0[i];
		}*/
		
		/*for(int i = 0; i < 10; ++i){
			System.out.println("Conditional dependency of " + i + " being 0 given a 0: " + w1probs0[i]);
			System.out.println("Conditional dependency of " + i + " being 0 given a 1: " + w1probs1[i]);
		}*/
		
				
		//test w1:
		System.out.println("\nDEPENDENT CLASSIFICATION\n");
		
		for(ADSamplePoint p: testing){
			if(p.theClass == 1){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				float p4 = 1.0f;
				
				rootStart = tree.rootval;
				if(p.features[rootStart] == 0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					p4 *= w4probs0[rootStart];
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
					p4 *= w4probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						p4 *= w4probs0[e.dest];
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						p4 *= w4probs1[e.dest];
					} 
				}
				
				if(p1 >= p2 && p1 >= p3 && p1 >= p4){
					p.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
					p.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
					p.classifierGuess = 3;
				} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
					p.classifierGuess = 4;
				} else p.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[p.theClass -1][p.classifierGuess -1] ++;
			}
			
			//test w2
			else if(p.theClass == 2){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				float p4 = 1.0f;
				
				rootStart = tree.rootval;
				if(p.features[rootStart] ==0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					p4 *= w4probs0[rootStart];
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
					p4 *= w4probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						p4 *= w4probs0[e.dest];
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						p4 *= w4probs1[e.dest];
					} 
				}
				
				if(p1 >= p2 && p1 >= p3 && p1 >= p4){
					p.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
					p.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
					p.classifierGuess = 3;
				} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
					p.classifierGuess = 4;
				} else p.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[p.theClass -1][p.classifierGuess -1] ++;
			}
			
			
			//test w3
			else if(p.theClass ==3){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				float p4 = 1.0f;
				
				rootStart = tree.rootval;
				if(p.features[rootStart] ==0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					p4 *= w4probs0[rootStart];
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
					p4 *= w4probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						p4 *= w4probs0[e.dest];
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						p4 *= w4probs1[e.dest];
					} 
				}
				
				if(p1 >= p2 && p1 >= p3 && p1 >= p4){
					p.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
					p.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
					p.classifierGuess = 3;
				} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
					p.classifierGuess = 4;
				} else p.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[p.theClass -1][p.classifierGuess -1] ++;
			}
			
			//test w4
			else if(p.theClass == 4){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				float p4 = 1.0f;
				
				rootStart = tree.rootval;
				if(p.features[rootStart] ==0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					p4 *= w4probs0[rootStart];
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
					p4 *= w4probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(p.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						p4 *= w4probs0[e.dest];
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						p4 *= w4probs1[e.dest];
					} 
				}
				
				if(p1 >= p2 && p1 >= p3 && p1 >= p4){
					p.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
					p.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
					p.classifierGuess = 3;
				} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
					p.classifierGuess = 4;
				} else p.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[p.theClass -1][p.classifierGuess -1] ++;
			}
		}
		
		System.out.println("\n..................\nConfusion matrix:");
		for(float[]i: confusionDependent){
			System.out.print("\n| ");
			for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%2.3f", 100* (0.0f+j)/400) + "% |");
			}			
		}
		
		
		
		System.out.println("\n..................");
		
		
	}
	
	public void decisionTree(){
		
		sortForTraining(FIVEFOLD_SET);
		ArrayList<Integer> allColumns = new ArrayList<>();
		for(int i = 0; i < NUMFEATURES; ++i)
			allColumns.add(i);
		
		ArrayList<DTStep> treeSteps = new ArrayList<DTStep>();
		
		DTStep treeRoot = getDTStep(allColumns, training, null);
		
		//test the testing sample
		for(ADSamplePoint p: testing){
			treeRoot.eval(p);
			try{confusionDecision[p.theClass -1][p.classifierGuess -1] ++;} catch( ArrayIndexOutOfBoundsException e){System.out.println(p);}
		}
		
		printTree(0, treeRoot);
		
		System.out.println("\n..................\nConfusion matrix:");
		for(float[]i: confusionDecision){
			System.out.print("\n| ");
			for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%2.3f", 100 *(0.0f + j)/400 ) + "% |");
			}			
		}
		
		System.out.println("\n..................");
			
	}
		
	
	
	public DTStep getDTStep(ArrayList<Integer> columns, ArrayList<ADSamplePoint> data, DTStep par){
		double best_entropy = blog(4.0);
		int best_column = -1;
		
		//calculate number of samples per class
		int[] classCounts = new int[5];
		for(ADSamplePoint p: data){
			switch(p.theClass){
			case 1: classCounts[1]++; break;
			case 2: classCounts[2]++; break;
			case 3: classCounts[3]++; break;
			case 4: classCounts[4]++; break;
			}
		}
//System.out.println("Class counts: " + Arrays.toString(classCounts));		
			
		//this gives the starting entropy
		double entropy = 0;		
		for(int i = 1; i < 5; ++i){
			double p = classCounts[i]/data.size();
			if(!Double.isNaN(-1 * p * blog(p)))
				entropy += -1 * p * blog(p);					
		}
		//System.out.println("entropy -- " + entropy);
		
		int[] i0Count = new int[5];
		int[] i1Count = new int[5];
		//this finds the column that gives the best information gain
		
		
		for(Integer i: columns){
			double entropyColi = 0.0;
			for(int j = 1; j < 5; ++j){
				for(ADSamplePoint p: training){
					//entropy i 0
					//i0Count = 0;
					//i1Count = 0;
					if(p.theClass == j){
						if(p.features[i] == 0){
							i0Count[j]++;
						} else if(p.features[i] == 1){
							i1Count[j]++;
						}						
					} //end if
				}//end for

			}
			//Information gain for this column is:
			//Total entropy - [for each class] proportion of this class over the samples * entropy of the class
			double ent = 0.0;
			for(int j = 1; j < 5; ++j){
				double prop = (0.0 + i0Count[j]) /(i1Count[j] + i0Count[j]);
				ent += -1 * prop * blog(prop);
			}
			entropyColi = entropy - ent;
			if(entropyColi < best_entropy){
				best_entropy = entropyColi;
				best_column = i;
			}
			
		}
		
		if(columns.size() == 1){
			best_column = columns.get(0);
		}
		
		DTStep theStep = new DTStep(0, 0, best_column, par);

		ArrayList<ADSamplePoint> posVals = new ArrayList<>();
		ArrayList<ADSamplePoint> negVals = new ArrayList<>();
		ArrayList<Integer> newCols= new ArrayList<>();
		
		if(columns.isEmpty()){
			theStep.rand = true;
			theStep.rvs = classCounts;
			return theStep;
		}
		
		
		for(Integer i: columns)
			if(i != best_column)
				newCols.add(i);
		for(ADSamplePoint dp: data){
			if(dp.features[best_column] == 1){
				posVals.add(dp);
			} else if(dp.features[best_column] ==0){
				negVals.add(dp);
			}
		}
		
		//check if the column gave definite class for 1 -- set classIfTrue
		//check if the column gave a definite class for 0 -- set classIfFalse
		boolean all1 = true;
		boolean all2 = true;
		boolean all3 = true;
		boolean all4 = true;
		int ct1 = 0;
		int ct2 = 0;
		int ct3 = 0;
		int ct4 = 0; //changed from 0
				
		if(posVals.size() > 0){
			for(ADSamplePoint dp: posVals){
				switch(dp.theClass){
				case 1:
					all2 = false;
					all3 = false;
					all4 = false;
					ct1++;
					break;
				case 2:
					all1 = false;
					all3 = false;
					all4 = false;
					ct2++;
					break;
				case 3:
					all2 = false;
					all1 = false;
					all4 = false;
					ct3++;
					break;
				case 4:
					all2 = false;
					all3 = false;
					all4 = false;
					ct4++;
					break;
				}
			}
			if(all1 || (0.0 + ct1) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfTrue = 1;
			else if(all2 || (0.0 + ct2) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfTrue = 2;
			else if(all3 || (0.0 + ct3) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfTrue = 3;
			else if(all4 || (0.0 + ct4) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfTrue = 4;
			
			if(data.size() == 0){
				if(ct1 > ct2 && ct1 > ct3 && ct1 > ct4) theStep.classIfTrue = 1;
				else if(ct2 > ct1 && ct2 > ct3 && ct2 > ct4) theStep.classIfTrue = 2;
				else if(ct3 > ct2 && ct3 > ct1 && ct3 > ct4) theStep.classIfTrue = 3;
				else if(ct4 > ct2 && ct4 > ct3 && ct4 > ct1) theStep.classIfTrue = 4;
			} /////////////////////////////////////// HERE
		}
		
		all1 = true;
		all2 = true;
		all3 = true;
		all4 = true;
		ct1 = 0;
		ct2 = 0;
		ct3 = 0;
		ct4 = 0; //changed from 0
		if(negVals.size() > 0){
			for(ADSamplePoint dp: negVals){
				switch(dp.theClass){
				case 1:
					all2 = false;
					all3 = false;
					all4 = false;
					ct1++;
					break;
				case 2:
					all1 = false;
					all3 = false;
					all4 = false;
					ct2++;
					break;
				case 3:
					all2 = false;
					all1 = false;
					all4 = false;
					ct3++;
					break;
				case 4:
					all2 = false;
					all3 = false;
					all4 = false;
					ct4++;
					break;
				}
			}
			if(all1|| (0.0 + ct1) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfFalse = 1;
			else if(all2|| (0.0 + ct2) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfFalse = 2;
			else if(all3|| (0.0 + ct3) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfFalse = 3;
			else if(all4|| (0.0 + ct4) / (ct1 + ct2 + ct3 + ct4) > DT_THRESH) theStep.classIfFalse = 4;
			
			if(data.size() == 0){
				if(ct1 > ct2 && ct1 > ct3 && ct1 > ct4) theStep.classIfFalse = 1;
				else if(ct2 > ct1 && ct2 > ct3 && ct2 > ct4) theStep.classIfFalse = 2;
				else if(ct3 > ct2 && ct3 > ct1 && ct3 > ct4) theStep.classIfFalse = 3;
				else if(ct4 > ct2 && ct4 > ct3 && ct4 > ct1) theStep.classIfFalse = 4;
			} 
		}
		
		if(theStep.classIfFalse != DTStep.VOID_DATA && theStep.classIfTrue != DTStep.VOID_DATA) return theStep;
		
//		System.out.println("New step (no children yet) " +theStep);
		
		if(theStep.classIfTrue == DTStep.VOID_DATA && posVals.size() > 0) theStep.dtStepIfTrue = getDTStep(newCols, posVals, theStep);
		if(theStep.classIfFalse == DTStep.VOID_DATA && negVals.size() > 0) theStep.dtStepIfFalse = getDTStep(newCols, negVals, theStep);
		
		//System.out.println(theStep);
		
		return theStep;
	}
	
	public void printTree(int depth, DTStep d){
		
		System.out.print(depth + " ");
		System.out.println(d);
		if(d.dtStepIfFalse != null){
			for(int i = 0; i < depth; ++i) System.out.print("..");
			System.out.print(" 0 ->");
			printTree(depth +1, d.dtStepIfFalse);
		}
		if(d.dtStepIfTrue != null){
			for(int i = 0; i < depth; ++i) System.out.print("..");
			System.out.print(" 1 ->");
			printTree(depth +1, d.dtStepIfTrue);
		}
	}
	

	public void getData(){
		/////////////////// Set up database
		//Connect to database
		try {
	
			//direct java to the sqlite-jdbc driver jar code
			// load the sqlite-JDBC driver using the current class loader
			Class.forName("org.sqlite.JDBC");
			System.out.println("Open Database Connection");
	
			//HARD CODED DATABASE NAME:
			database = DriverManager.getConnection("jdbc:sqlite:a3");
		     //create a statement object which will be used to relay a
		     //sql query to the database
			stat = database.createStatement();
		
	
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		//go do some stuff with the DB
		
		ADSamplePoint pt;
			
		try{
			String sqlQueryString = "SELECT * FROM artificialData";
			//System.out.println(sqlQueryString);			
			ResultSet rs = stat.executeQuery(sqlQueryString);
			 
	        while (rs.next()) {
	        	pt = new ADSamplePoint(rs.getInt("class"), rs.getInt("w1"), rs.getInt("w2"),rs.getInt("w3"),rs.getInt("w4"),rs.getInt("w5"),rs.getInt("w6"),rs.getInt("w7"),rs.getInt("w8"),rs.getInt("w9"),rs.getInt("w10"));
	        	data.add(pt);
	        }
			
		}
		catch(SQLException e){
			e.printStackTrace();			
		}
			

		/*for(SamplePoint p: data){
			print(p);
		}*/
		
		//System.out.println("" + data.size() + " samples");
		
		
	}
	
	public void estimateDependenceTree(){
		//find the mutual information measure for all <i,j>
		//for all samples:
		//for all<i,j>:
			//sum <0,0> + <0,1> + <1,0> + <1,1> for PR(v1=0, v4=0) log(pr(1=0, 4=0) /Pr(1=0)(Pr4=0)
		
		//find the probability of a zero in the sample set
		float[] probNequals0 = new float[10];
		float[][] probIJequals00 = new float[10][10];
		float[][] probIJequals01 = new float[10][10];
		float[][] probIJequals10 = new float[10][10];
		float[][] probIJequals11 = new float[10][10];
		
		mim = new float[10][10];
		
		for(int i = 0; i < 10; ++i) for(int j = 0; j < 10; ++j){
			probIJequals00[i][j] = 0;
			probIJequals01[i][j] = 0;
			probIJequals10[i][j] = 0;
			probIJequals11[i][j] = 0;			
			mim[i][j] = 0;
		}
		
		for(ADSamplePoint p: data){
			for(int i = 0; i < 10; ++i){
				probNequals0[i] += p.features[i];
			}
			
			for(int i= 0; i < 10; ++i) for(int j = 0; j < 10; ++j){
				if(p.features[i] == 0 && p.features[j] == 0 ) probIJequals00[i][j]++;
				else if (p.features[i] ==0 && p.features[j] == 1 ) probIJequals01[i][j]++;
				else if (p.features[i] == 1 && p.features[j] == 0 ) probIJequals10[i][j]++;
				else if (p.features[i] ==  1 && p.features[j] ==1 ) probIJequals11[i][j]++;
			}
		}
		
		//change this to probability of zero and divide by sample size
		for(int i = 0; i < 10; ++i){
			probNequals0[i] /= data.size();
			probNequals0[i] = 1 - probNequals0[i];
			
			//System.out.println("Prob of " + i + " = 0 : " + probNequals0[i]);
			//System.out.println("Prob of " + i + " = 1 : " + (1-probNequals0[i]));
		}
		
		for(int i= 0; i < 10; ++i) for(int j = 0; j < 10; ++j){
			probIJequals00[i][j] /= data.size();
			probIJequals01[i][j] /= data.size();
			probIJequals10[i][j] /= data.size();
			probIJequals11[i][j] /= data.size();
			
			if(Float.isNaN(probIJequals00[i][j]) ) probIJequals00[i][j] = 0;//Float.MIN_VALUE;
			if(Float.isNaN(probIJequals01[i][j]) ) probIJequals01[i][j] = 0;//Float.MIN_VALUE;
			if(Float.isNaN(probIJequals10[i][j]) ) probIJequals10[i][j] = 0;//Float.MIN_VALUE;
			if(Float.isNaN(probIJequals11[i][j]) ) probIJequals11[i][j] = 0;//Float.MIN_VALUE;
			
			
		}
		
		//calculate mutual information measure
		for(int i = 0; i < 10; ++i) for(int j = 0; j < 10; ++j){
			mim[i][j] =  (float) (probIJequals00[i][j] * (Math.log(probIJequals00[i][j] / (probNequals0[i] * probNequals0[j]))))
					+ (float) (probIJequals01[i][j] * (Math.log(probIJequals01[i][j] / (probNequals0[i] * (1-probNequals0[j])))))
					+ (float) (probIJequals10[i][j] * (Math.log(probIJequals10[i][j] / ((1-probNequals0[i]) * probNequals0[j]))))
					+ (float) (probIJequals11[i][j] * (Math.log(probIJequals11[i][j] / ((1-probNequals0[i]) * (1-probNequals0[j])))))
					;
			if(Float.isNaN(mim[i][j])) mim[i][j] = 0;
			/*System.out.println(".........");
			System.out.println("Prob of " + i + " = 0 : " + probNequals0[i]);
			System.out.println("Prob of " + i + " = 1 : " + (1-probNequals0[i]));
			
			System.out.println("Prob of " + i + "= 0 ^ " + j + " = 0 :" +  probIJequals00[i][j]);
			System.out.println("Prob of " + i + "= 0 ^ " + j + " = 1 :" +  probIJequals01[i][j]);
			System.out.println("Prob of " + i + "= 1 ^ " + j + " = 0 :" +  probIJequals10[i][j]);
			System.out.println("Prob of " + i + "= 1 ^ " + j + " = 1 :" +  probIJequals11[i][j]);

			System.out.println("MIM [" + i + "][" + j + "] :" + mim[i][j]);*/
		}
		

		getTree();

	}
	
	public void getTree(){
		tree = new KruskalTree(10, 45, mim);
		tree.graph.KruskalMST();
	}
	
	public void print(ADSamplePoint p){
		String ret = ("Class: " + p.theClass + " " + "   Classifier guess: " + p.classifierGuess);
		for(int i: p.features){
			ret += "[" + i + "] ";
		}
		System.out.println(ret);
	}
	
	public void sortForTraining(int set){
			
		training.clear();
		testing.clear();
		for(int i = 0; i < data.size(); ++i){
			if(i % 5 == set){
				testing.add(data.get(i));
			} else training.add(data.get(i));
		}
	}
	
	public double blog(double x){
		return Math.log(x) / Math.log(2);
	}
}


/* SCRAP
 * 		//test w2
		/*
		for(int i = 3600; i < 4000; ++i){
			float p1 = 1.0f;
			float p2 = 1.0f;
			float p3 = 1.0f;
			float p4 = 1.0f;
			
			for(int j = 0; j < 10; ++j){
				if(data.get(i).features[j] == 1) p1 *= w1probs[j];
				else p1 *= (1-w1probs[j]);
				if(data.get(i).features[j] == 1) p2 *= w2probs[j];
				else p2 *= (1-w2probs[j]);
				if(data.get(i).features[j] == 1) p3 *= w3probs[j];
				else p3 *= (1-w3probs[j]);
				if(data.get(i).features[j] == 1) p4 *= w2probs[j];
				else p4 *= (1-w4probs[j]);
			}
			
			if(p1 >= p2 && p1 >= p3 && p1 >= p4){
				data.get(i).classifierGuess = 1;
			} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
				data.get(i).classifierGuess = 2;
			} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
				data.get(i).classifierGuess = 3;
			} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
				data.get(i).classifierGuess = 4;
			} else data.get(i).classifierGuess = 9999;
			
			//print(data.get(i));
			confusionIndependent[data.get(i).theClass -1][data.get(i).classifierGuess -1] ++;
		}
		
		//test w3
		
		for(int i = 5600; i < 6000; ++i){
			float p1 = 1.0f;
			float p2 = 1.0f;
			float p3 = 1.0f;
			float p4 = 1.0f;
			
			for(int j = 0; j < 10; ++j){
				if(data.get(i).features[j] == 1) p1 *= w1probs[j];
				else p1 *= (1-w1probs[j]);
				if(data.get(i).features[j] == 1) p2 *= w2probs[j];
				else p2 *= (1-w2probs[j]);
				if(data.get(i).features[j] == 1) p3 *= w3probs[j];
				else p3 *= (1-w3probs[j]);
				if(data.get(i).features[j] == 1) p4 *= w2probs[j];
				else p4 *= (1-w4probs[j]);
			}
			
			if(p1 >= p2 && p1 >= p3 && p1 >= p4){
				data.get(i).classifierGuess = 1;
			} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
				data.get(i).classifierGuess = 2;
			} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
				data.get(i).classifierGuess = 3;
			} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
				data.get(i).classifierGuess = 4;
			} else data.get(i).classifierGuess = 9999;
			
			//print(data.get(i));
			confusionIndependent[data.get(i).theClass -1][data.get(i).classifierGuess -1] ++;
		}
		
		//test w4
		
		for(int i = 7600; i < 8000; ++i){
			float p1 = 1.0f;
			float p2 = 1.0f;
			float p3 = 1.0f;
			float p4 = 1.0f;
			
			for(int j = 0; j < 10; ++j){
				if(data.get(i).features[j] == 1) p1 *= w1probs[j];
				else p1 *= (1-w1probs[j]);
				if(data.get(i).features[j] == 1) p2 *= w2probs[j];
				else p2 *= (1-w2probs[j]);
				if(data.get(i).features[j] == 1) p3 *= w3probs[j];
				else p3 *= (1-w3probs[j]);
				if(data.get(i).features[j] == 1) p4 *= w2probs[j];
				else p4 *= (1-w4probs[j]);
			}
			
			if(p1 >= p2 && p1 >= p3 && p1 >= p4){
				data.get(i).classifierGuess = 1;
			} else if(p2 >= p1 && p2 >= p3 && p2 >= p4){
				data.get(i).classifierGuess = 2;
			} else if(p3 >= p1 && p3 >= p2 && p3 >= p4){
				data.get(i).classifierGuess = 3;
			} else if(p4 >= p1 && p4 >= p2 && p4 >= p3){
				data.get(i).classifierGuess = 4;
			} else data.get(i).classifierGuess = 9999;
			
			//print(data.get(i));
			confusionIndependent[data.get(i).theClass -1][data.get(i).classifierGuess -1] ++;
		}*/
