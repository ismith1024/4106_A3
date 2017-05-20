package classifier;



import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import classifier.KruskalTree.Graph.Edge;

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

public class WineClassifier {
	
	static final int FIVEFOLD_SET = 4;

	
	
	static Connection database;
	static Statement stat;
	
	static final int NUMSAMPLES = 178;
	static final int NUMFEATURES = 13;
	static final int NUMCLASSES = 3;
	
	static int w1_count;
	static int w2_count;
	static int w3_count;
	
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
	
	static ArrayList<WineSamplePoint> data;
	static ArrayList<WineSamplePoint> training;
	ArrayList<WineSamplePoint> testing;
	
	float[][] mim;
	
	KruskalTree tree;
	
	public WineClassifier(){
		data = new ArrayList<>();
		training = new ArrayList<>();
		testing = new ArrayList<>();
		
		w1vals = new float[NUMFEATURES][5];
		w2vals = new float[NUMFEATURES][5];
		w3vals = new float[NUMFEATURES][5];
		w4vals = new float[NUMFEATURES][5];
		
		confusion1 = new float[NUMFEATURES][NUMFEATURES];
		confusion2 = new float[NUMFEATURES][NUMFEATURES];
		confusion3 = new float[NUMFEATURES][NUMFEATURES];
		confusion4 = new float[NUMFEATURES][NUMFEATURES];
		
		confusionIndependent = new float[NUMCLASSES][NUMCLASSES];	
		confusionDependent = new float[NUMCLASSES][NUMCLASSES];
		confusionDecision = new float[NUMCLASSES][NUMCLASSES];

	}
	
	public static void main(String[] args) {
		
		WineClassifier classifier = new WineClassifier();
		
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
		//System.out.println("Learn the dependence tree topology:");
		estimateDependenceTree();
		
		System.out.println("===================================\nBayesian Dependence Tree Classification:");
		//dependent bayesian classification
		dependentBayes();
		
		

		
		
	}
		
	public void independentBayes() {
		//five-fold cross validation
		//will train with samples 0-1599, 2000-3599, 4000-5599, 6000-7599
		//will test with samples 1600-1999, 3600-3699, 5600-5699, 7600-7699
		float[] w1probs = new float[NUMFEATURES];
		float[] w2probs = new float[NUMFEATURES];
		float[] w3probs = new float[NUMFEATURES];
		
		for(float[]i: confusionIndependent) for(float j: i) j = 0;
		
		training = new ArrayList<>();
		testing = new ArrayList<>();
		
		sortForTraining(FIVEFOLD_SET);
		
		int training1 = 0;
		int training2 = 0;
		int training3 = 0;
		
		for(WineSamplePoint sample: training){
			if(sample.theClass == 1){
				for(int j = 0; j < NUMFEATURES; ++j) w1probs[j] += sample.features[j];
				training1++;
			}
			else if(sample.theClass == 2){ 
				for(int j = 0; j < NUMFEATURES; ++j) w2probs[j] += sample.features[j];
				training2++;
			}
			else{ 
				for(int j = 0; j < NUMFEATURES; ++j) w3probs[j] += sample.features[j];
				training3++;
			}
		}		
		
		for(int i = 0; i < NUMFEATURES; ++i){
			w1probs[i] /= training1;
			w2probs[i] /= training2;
			w3probs[i] /= training3;
		}
		
		//test w1:
		for(WineSamplePoint sample: testing){
			float p1 = 1.0f;
			float p2 = 1.0f;
			float p3 = 1.0f;
			
			for(int j = 0; j < NUMFEATURES; ++j){
				if(sample.features[j] == 1) p1 *= w1probs[j];
				else p1 *= (1-w1probs[j]);
				if(sample.features[j] == 1) p2 *= w2probs[j];
				else p2 *= (1-w2probs[j]);
				if(sample.features[j] == 1) p3 *= w3probs[j];
				else p3 *= (1-w3probs[j]);
			}
			
			if(p1 >= p2 && p1 >= p3){
				sample.classifierGuess = 1;
			} else if(p2 >= p1 && p2 >= p3){
				sample.classifierGuess = 2;
			} else if(p3 >= p1 && p3 >= p2){
				sample.classifierGuess = 3;
			} else sample.classifierGuess = 9999;
			
			//print(sample);
			confusionIndependent[sample.theClass -1][sample.classifierGuess -1] ++;
	
		}
		
		int[] wct = new int[NUMCLASSES];
		wct[0] = 0;
		wct[1] = 0;
		wct[2] = 0;
		for(WineSamplePoint p: testing){
			wct[p.theClass - 1]++;
		}
		
		System.out.println("\n..................\nConfusion matrix:");
		for(int i = 0; i < NUMCLASSES; ++i){ //for(float[]i: confusionDependent){
			System.out.print("\n| ");
			for(int j = 0; j < NUMCLASSES; ++j){ //for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%3.2f", (0.0f + 100.00 * confusionIndependent[i][j])/wct[i]) + "% |");
				//System.out.print("" + j + " |");
			}			
		}
				
		/*System.out.println("\n..................\nConfusion matrix:");
		for(float[]i: confusionIndependent){
			System.out.print("\n| ");
			for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%2.3f", 100 * (0.0f + j)/1600) + "% |");
			}			
		}*/
		
		System.out.println("\n..................");
		
	}
	
	public void dependentBayes(){
		//five-fold cross validation
		//will train with samples 0-1599, 2000-3599, 4000-5599, 6000-7599
		//will test with samples 1600-1999, 3600-3699, 5600-5699, 7600-7699
		
		//These represent w1probs0[n] = probability that value n is true given its parent is 0
		float[] w1probs0 = new float[NUMFEATURES];
		float[] w2probs0 = new float[NUMFEATURES];
		float[] w3probs0 = new float[NUMFEATURES];
		float[] w1probs1 = new float[NUMFEATURES];
		float[] w2probs1 = new float[NUMFEATURES];
		float[] w3probs1 = new float[NUMFEATURES];
		
		//counts the number of zeros in each feature
		float[] w1Count = new float[NUMFEATURES];
		float[] w2Count = new float[NUMFEATURES];
		float[] w3Count = new float[NUMFEATURES];
		
		for(float[]i: confusionDependent) for(float j: i) j = 0;
		
		sortForTraining(FIVEFOLD_SET);
		
		int training1 = 0;
		int training2 = 0;
		int training3 = 0;

		int rootStart = tree.rootval;
		
		for(WineSamplePoint sample: training){
			if(sample.theClass == 1){

				w1probs0[rootStart] += sample.features[rootStart];
				w1probs1[rootStart] += sample.features[rootStart];
				w1Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0)
						w1probs0[e.dest]++;// += sample.features[e.dest];
					else w1probs1[e.dest]++;// += sample.features[e.dest];
					w1Count[e.dest]++;
				}			
								
				training1++;
			}
			else if(sample.theClass == 2){ 

				//train w2
				w2probs0[rootStart] += sample.features[rootStart];
				w2probs1[rootStart] += sample.features[rootStart];
				w2Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0)
						w2probs0[e.dest]++;// += sample.features[e.dest];
					else w2probs1[e.dest]++;// += sample.features[e.dest];
					w2Count[e.dest]++;
				}
			
				training2++;
			}
			else{ 

				//train w3
				w3probs0[rootStart] += sample.features[rootStart];
				w3probs1[rootStart] += sample.features[rootStart];
				w3Count[rootStart]++;
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0)
						w3probs0[e.dest]++; //+= sample.features[e.dest];
					else w3probs1[e.dest]++; //+= sample.features[e.dest];
					w3Count[e.dest]++;
				}
				
				training3++;
			}
		}
		

		/*for(int i = 0; i < NUMFEATURES; ++i){
			w1probs0[i] = 1 - w1probs0[i];
			w2probs0[i] = 1 - w2probs0[i];
			w3probs0[i] = 1 - w3probs0[i];
		}*/
		for(int i = 0; i < NUMFEATURES; ++i){
			w1probs0[i] /= training1;
			w1probs1[i] /= training1;
			w2probs0[i] /= training2;
			w2probs1[i] /= training2;
			w3probs0[i] /= training3;
			w3probs1[i] /= training3;
		

			//System.out.println("Conditional dependency of " + i + " being 0 given a 0: " + w1probs0[i]);
			//System.out.println("Conditional dependency of " + i + " being 0 given a 1: " + w1probs1[i]);
		}
		
		w1probs0[rootStart] = 1 - w1probs0[rootStart];
		w2probs0[rootStart] = 1 - w2probs0[rootStart];
		w3probs0[rootStart] = 1 - w3probs0[rootStart];
		
		
		//test w1:
		System.out.println("\nDEPENDENT CLASSIFICATION\n");
		
		for(WineSamplePoint sample: testing){
		
			if(sample.theClass == 1){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				
				rootStart = tree.rootval;
				if(sample.features[rootStart] == 0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
//System.out.printf("Root: wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n",w1probs0[rootStart],w2probs0[rootStart],w3probs0[rootStart] ,p1, p2, p3);						
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
//System.out.printf("Root: wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", w1probs1[rootStart],w2probs1[rootStart],w3probs1[rootStart], p1, p2, p3);	
				}
				
			
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
//System.out.printf("%d--%d wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs0[e.dest],w2probs0[e.dest],w3probs0[e.dest], p1, p2, p3);
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
//System.out.printf("%d--%d wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs1[e.dest],w2probs1[e.dest],w3probs1[e.dest], p1, p2, p3);
					} 
				}
				
				//TODO: Multiply by number of instances per class
				if(p1 >= p2 && p1 >= p3){
					sample.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3){
					sample.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2){
					sample.classifierGuess = 3;
				} else sample.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[sample.theClass -1][sample.classifierGuess -1] ++;
			}
			
			//test w2
			else if(sample.theClass == 2){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				
				rootStart = tree.rootval;
				if(sample.features[rootStart] ==0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						//System.out.printf("%d--%d wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs0[e.dest],w2probs0[e.dest],w3probs0[e.dest], p1, p2, p3);
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						//System.out.printf("%d--%d wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs1[e.dest],w2probs1[e.dest],w3probs1[e.dest], p1, p2, p3);
					} 
				}
				
				if(p1 >= p2 && p1 >= p3){
					sample.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3){
					sample.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2){
					sample.classifierGuess = 3;
				} else sample.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[sample.theClass -1][sample.classifierGuess -1] ++;
			}		
			
			//test w3
			else if(sample.theClass == 3){
				float p1 = 1.0f;
				float p2 = 1.0f;
				float p3 = 1.0f;
				
				rootStart = tree.rootval;
				if(sample.features[rootStart] ==0 ){
					p1 *= w1probs0[rootStart];
					p2 *= w2probs0[rootStart];
					p3 *= w3probs0[rootStart];
					
				} else{
					p1 *= w1probs1[rootStart];
					p2 *= w2probs1[rootStart];
					p3 *= w3probs1[rootStart];
				}
				
				for(Edge e: tree.edges){
					if(sample.features[e.src] == 0){
						p1 *= w1probs0[e.dest];
						p2 *= w2probs0[e.dest];
						p3 *= w3probs0[e.dest];
						//System.out.printf("%d--%d wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs0[e.dest],w2probs0[e.dest],w3probs0[e.dest], p1, p2, p3);
					} else {
						p1 *= w1probs1[e.dest];
						p2 *= w2probs1[e.dest];
						p3 *= w3probs1[e.dest];
						//System.out.printf("%d--%d: wt1: %2.3f wt2: %2.3f wt3: %2.3f  total after mult -- p1: %2.3f p2: %2.3f p3: %2.3f\n", e.src, e.dest, w1probs1[e.dest],w2probs1[e.dest],w3probs1[e.dest], p1, p2, p3);
					} 
				}
				
				if(p1 >= p2 && p1 >= p3){
					sample.classifierGuess = 1;
				} else if(p2 >= p1 && p2 >= p3){
					sample.classifierGuess = 2;
				} else if(p3 >= p1 && p3 >= p2){
					sample.classifierGuess = 3;
				} else sample.classifierGuess = 9999;
				
				//print(data.get(i));
				confusionDependent[sample.theClass -1][sample.classifierGuess -1] ++;
			}
		}
		
		int[] wct = new int[NUMCLASSES];
		wct[0] = 0;
		wct[1] = 0;
		wct[2] = 0;
		for(WineSamplePoint p: testing){
			wct[p.theClass - 1]++;
		}
		
		System.out.println("\n..................\nConfusion matrix:");
		for(int i = 0; i < NUMCLASSES; ++i){ //for(float[]i: confusionDependent){
			System.out.print("\n| ");
			for(int j = 0; j < NUMCLASSES; ++j){ //for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%3.2f", (0.0f + 100.00 * confusionDependent[i][j])/wct[i]) + "% |");
				//System.out.print("" + j + " |");
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
		for(WineSamplePoint p: testing){
			treeRoot.eval(p);
			System.out.println("CL: " + (p.theClass -1) + "guess: " + (p.classifierGuess -1));
			/*try{*/confusionDecision[p.theClass -1][p.classifierGuess -1] ++;/*} catch( ArrayIndexOutOfBoundsException e){System.out.println(p);}*/
		}
		
		int[] wct = new int[NUMCLASSES];
		wct[0] = 0;
		wct[1] = 0;
		wct[2] = 0;
		for(WineSamplePoint p: testing){
			wct[p.theClass - 1]++;
		}
		
		printTree(0, treeRoot);
		
		System.out.println("\n..................\nConfusion matrix:");
		for(int i = 0; i < NUMCLASSES; ++i){ //for(float[]i: confusionDependent){
			System.out.print("\n| ");
			for(int j = 0; j < NUMCLASSES; ++j){ //for(float j: i){
				//j /= 500;
				System.out.print("" + String.format("%3.2f", (0.0f + 100.00 * confusionDecision[i][j])/wct[i]) + "% |");
				//System.out.print("" + j + " |");
			}			
		}
		
		System.out.println("\n..................");
			
	}
	
	public DTStep getDTStep(ArrayList<Integer> columns, ArrayList<WineSamplePoint> data, DTStep par){
		double best_entropy = blog(4.0);
		int best_column = -1;
		
		//calculate number of samples per class
		int[] classCounts = new int[5];
		for(WineSamplePoint p: data){
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
		for(int i = 1; i < 4; ++i){
			double p = classCounts[i]/data.size();
			if(!Double.isNaN(-1 * p * blog(p)))
				entropy += -1 * p * blog(p);					
		}
		//System.out.println("entropy -- " + entropy);
		
		int[] i0Count = new int[4];
		int[] i1Count = new int[4];
		//this finds the column that gives the best information gain
		
		
		for(Integer i: columns){
			double entropyColi = 0.0;
			for(int j = 1; j < 4; ++j){
				for(WineSamplePoint p: training){
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
			for(int j = 1; j < 4; ++j){
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

		ArrayList<WineSamplePoint> posVals = new ArrayList<>();
		ArrayList<WineSamplePoint> negVals = new ArrayList<>();
		ArrayList<Integer> newCols= new ArrayList<>();
		
		if(columns.isEmpty()){
			theStep.rand = true;
			theStep.rvs = classCounts;
			return theStep;
		}
		
		
		for(Integer i: columns)
			if(i != best_column)
				newCols.add(i);
		for(WineSamplePoint dp: data){
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
			for(WineSamplePoint dp: posVals){
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
			for(WineSamplePoint dp: negVals){
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
		
		//System.out.println("New step (no children yet) " +theStep);
		
		if(theStep.classIfTrue == DTStep.VOID_DATA && posVals.size() > 0) theStep.dtStepIfTrue = getDTStep(newCols, posVals, theStep);
		if(theStep.classIfFalse == DTStep.VOID_DATA && negVals.size() > 0) theStep.dtStepIfFalse = getDTStep(newCols, negVals, theStep);
		
		//System.out.println("Assigned children - return from recursion " + theStep);
		
		return theStep;
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
		
		WineSamplePoint pt;
			
		try{
			String sqlQueryString = "SELECT * FROM wines";
			//System.out.println(sqlQueryString);			
			ResultSet rs = stat.executeQuery(sqlQueryString);
			 
	        while (rs.next()) {
	        	pt = new WineSamplePoint(	        	
					rs.getInt("class"), rs.getInt("alcohol"), rs.getInt("malicAcid"), rs.getInt("ash"), rs.getInt("alkalinity"),
					rs.getInt("mg"), rs.getInt("phenols"), rs.getInt("flavanoids"), rs.getInt("nonPhenols"), rs.getInt("proanthos"),
					rs.getInt("intensity"), rs.getInt("hue"), rs.getInt("od280"), rs.getInt("proline"));
        	
	        	data.add(pt);
	        	if(pt.theClass == 1) w1_count++;
	        	else if(pt.theClass == 2) w2_count ++;
	        	else if(pt.theClass == 3) w3_count ++;
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
		float[] probNequals0 = new float[NUMFEATURES];
		float[][] probIJequals00 = new float[NUMFEATURES][NUMFEATURES];
		float[][] probIJequals01 = new float[NUMFEATURES][NUMFEATURES];
		float[][] probIJequals10 = new float[NUMFEATURES][NUMFEATURES];
		float[][] probIJequals11 = new float[NUMFEATURES][NUMFEATURES];
		
		mim = new float[NUMFEATURES][NUMFEATURES];
		
		for(int i = 0; i < NUMFEATURES; ++i) for(int j = 0; j < NUMFEATURES; ++j){
			probIJequals00[i][j] = 0;
			probIJequals01[i][j] = 0;
			probIJequals10[i][j] = 0;
			probIJequals11[i][j] = 0;			
			mim[i][j] = 0;
		}
		
		for(WineSamplePoint p: data){
			for(int i = 0; i < NUMFEATURES; ++i){
				probNequals0[i] += p.features[i];
			}
			
			for(int i= 0; i < NUMFEATURES; ++i) for(int j = 0; j < NUMFEATURES; ++j){
				if(p.features[i] == 0 && p.features[j] == 0 ) probIJequals00[i][j]++;
				else if (p.features[i] ==0 && p.features[j] == 1 ) probIJequals01[i][j]++;
				else if (p.features[i] == 1 && p.features[j] == 0 ) probIJequals10[i][j]++;
				else if (p.features[i] ==  1 && p.features[j] ==1 ) probIJequals11[i][j]++;
			}
		}
		
		//change this to probability of zero and divide by sample size
		for(int i = 0; i < NUMFEATURES; ++i){
			probNequals0[i] /= data.size();
			probNequals0[i] = 1 - probNequals0[i];
			
			/*System.out.println("Prob of " + i + " = 0 : " + probNequals0[i]);
			System.out.println("Prob of " + i + " = 1 : " + (1-probNequals0[i]));*/
		}
		
		for(int i= 0; i < NUMFEATURES; ++i) for(int j = 0; j < NUMFEATURES; ++j){
			probIJequals00[i][j] /= data.size();
			probIJequals01[i][j] /= data.size();
			probIJequals10[i][j] /= data.size();
			probIJequals11[i][j] /= data.size();
			
			if(Float.isNaN(probIJequals00[i][j]) ) probIJequals00[i][j] = 0;
			if(Float.isNaN(probIJequals01[i][j]) ) probIJequals01[i][j] = 0;
			if(Float.isNaN(probIJequals10[i][j]) ) probIJequals10[i][j] = 0;
			if(Float.isNaN(probIJequals11[i][j]) ) probIJequals11[i][j] = 0;
			
			//System.out.println("Prob of " + i + "= 0 ^ " + j + " = 0 :" +  probIJequals00[i][j]);
			//System.out.println("Prob of " + i + "= 0 ^ " + j + " = 1 :" +  probIJequals01[i][j]);
			//System.out.println("Prob of " + i + "= 1 ^ " + j + " = 0 :" +  probIJequals10[i][j]);
			//System.out.println("Prob of " + i + "= 1 ^ " + j + " = 1 :" +  probIJequals11[i][j]);			
		}
		
		//calculate mutual information measure
		for(int i = 0; i < NUMFEATURES; ++i) for(int j = 0; j < NUMFEATURES; ++j){
			mim[i][j] =  (float) (probIJequals00[i][j] * (Math.log(probIJequals00[i][j] / (probNequals0[i] * probNequals0[j]))))
					+  (float) (probIJequals01[i][j] * (Math.log(probIJequals01[i][j] / (probNequals0[i] * (1-probNequals0[j])))))
					+  (float) (probIJequals10[i][j] * (Math.log(probIJequals10[i][j] / ((1-probNequals0[i]) * probNequals0[j]))))
					+  (float) (probIJequals11[i][j] * (Math.log(probIJequals11[i][j] / ((1-probNequals0[i]) * (1-probNequals0[j])))))
					;
			
			if(Float.isNaN(mim[i][j])) mim[i][j] = 0;

			//System.out.println("MIM [" + i + "][" + j + "] :" + mim[i][j]);
		}

		
		getTree();

	}
	
	public void getTree(){
		tree = new KruskalTree(NUMFEATURES, (NUMFEATURES * (NUMFEATURES -1) /2), mim);
		tree.graph.KruskalMST();
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
	
	public void print(WineSamplePoint p){
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
