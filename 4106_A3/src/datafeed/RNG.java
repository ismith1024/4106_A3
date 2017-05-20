package datafeed;

import java.util.ArrayList;
import java.util.Random;

import classifier.ADSamplePoint;

public class RNG {

	/*
	         3
	         |
	 +-------+-----------+
	 |       |           |        
	 4       7	         2
	 |       |           |
	 5       9    +------+------+
	         |    |      |      |
	         8    6      10     1
	         
	*/
	static int[] ancestors = {2, 2, 3, -1, 3, 4, 2, 3, 8, 7};
	
	double[][] dependencies0;
	double[][] dependencies1;
	
	static int NUMCLASSES = 4;
	static int NUMFEATURES = 10;
	static int SAMPLES_PER_CLASS = 2000;
	
	ArrayList<ADSamplePoint> data;
	
	/*public class SamplePoint{
		int cl;
		int[] feat;
		
		public SamplePoint( int c){
			feat = new int[NUMFEATURES];
			cl = c;
		}
		
		@Override
		public String toString(){
			return String.format("Class: %d  Vals:[%d][%d][%d][%d][%d][%d][%d][%d][%d][%d]", cl, feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[0]); 
		}
	}*/
	
	public RNG(){
		dependencies0 = new double[NUMCLASSES][NUMFEATURES];
		dependencies1 = new double[NUMCLASSES][NUMFEATURES];
		data = new ArrayList<>();
	}

	public ArrayList<ADSamplePoint> get(){
		
		long seed = System.nanoTime();
		Random generator = new Random(seed);
		
		double aNum = generator.nextDouble();

		//set up the dependencies
		
		dependencies0[0][3] = 0.2;
		dependencies0[1][3] = 0.4;
		dependencies0[2][3] = 0.6;
		dependencies0[3][3] = 0.8;
		dependencies1[0][3] = 0.2;
		dependencies1[1][3] = 0.4;
		dependencies1[2][3] = 0.6;
		dependencies1[3][3] = 0.8;
				
		for(int i = 0; i < NUMCLASSES; ++i){
			for(int j = 0; j < NUMFEATURES; ++j){
				if(ancestors[j] != -1){
					if(j % (i+2) ==0){
						dependencies0[i][j] = 0.5 + generator.nextDouble() /2;
						dependencies1[i][j] = 0.5 - generator.nextDouble() /2;
					} else{
						dependencies0[i][j] = 0.5 - generator.nextDouble() /2;
						dependencies1[i][j] = 0.5 + generator.nextDouble() /2;
					}
				}
			}
		}
		
		//generate the data
		for(int i = 0; i < NUMCLASSES; ++i){
			for(int j = 0; j < SAMPLES_PER_CLASS; ++j){
				ADSamplePoint sam = new ADSamplePoint(i + 1);
				sam.features[3] = (generator.nextDouble() < dependencies0[i][3]) ? 0:1;
				//generate 4 from 3
				if(sam.features[3] == 0){
					sam.features[4] = (generator.nextDouble() < dependencies0[i][4]) ? 0:1;
				} else {
					sam.features[4] = (generator.nextDouble() < dependencies1[i][4]) ? 0:1;
				}
				
				//generate 7 from 3
				if(sam.features[3] == 0){
					sam.features[7] = (generator.nextDouble() < dependencies0[i][7]) ? 0:1;
				} else {
					sam.features[7] = (generator.nextDouble() < dependencies1[i][7]) ? 0:1;
				}
				
				//generate 2 from 3
				if(sam.features[3] == 0){
					sam.features[2] = (generator.nextDouble() < dependencies0[i][2]) ? 0:1;
				} else {
					sam.features[2] = (generator.nextDouble() < dependencies1[i][2]) ? 0:1;
				}
				
				//generate 5 from 4
				if(sam.features[4] == 0){
					sam.features[5] = (generator.nextDouble() < dependencies0[i][5]) ? 0:1;
				} else {
					sam.features[5] = (generator.nextDouble() < dependencies1[i][5]) ? 0:1;
				}
				
				//generate 9 from 7
				if(sam.features[7] == 0){
					sam.features[9] = (generator.nextDouble() < dependencies0[i][9]) ? 0:1;
				} else {
					sam.features[9] = (generator.nextDouble() < dependencies1[i][9]) ? 0:1;
				}
				
				//generate 8 from 9
				if(sam.features[9] == 0){
					sam.features[8] = (generator.nextDouble() < dependencies0[i][8]) ? 0:1;
				} else {
					sam.features[8] = (generator.nextDouble() < dependencies1[i][8]) ? 0:1;
				}
				
				//generate 6 from 2
				if(sam.features[2] == 0){
					sam.features[6] = (generator.nextDouble() < dependencies0[i][6]) ? 0:1;
				} else {
					sam.features[6] = (generator.nextDouble() < dependencies1[i][6]) ? 0:1;
				}
				
				//generate 10 from 2
				if(sam.features[2] == 0){
					sam.features[0] = (generator.nextDouble() < dependencies0[i][0]) ? 0:1;
				} else {
					sam.features[0] = (generator.nextDouble() < dependencies1[i][0]) ? 0:1;
				}
				
				//generate 1 from 2
				if(sam.features[2] == 0){
					sam.features[1] = (generator.nextDouble() < dependencies0[i][1]) ? 0:1;
				} else {
					sam.features[1] = (generator.nextDouble() < dependencies1[i][1]) ? 0:1;
				}
				
				data.add(sam);
			}

		}
		
		/*for(ADSamplePoint s: data){
			System.out.println(s);
		}*/
		
		return data;
		
	}
	
	public void run(){
	
		long seed = System.nanoTime();
		Random generator = new Random(seed);
		
		double aNum = generator.nextDouble();

		//set up the dependencies
		
		dependencies0[0][3] = 0.2;
		dependencies0[1][3] = 0.4;
		dependencies0[2][3] = 0.6;
		dependencies0[3][3] = 0.8;
		dependencies1[0][3] = 0.2;
		dependencies1[1][3] = 0.4;
		dependencies1[2][3] = 0.6;
		dependencies1[3][3] = 0.8;
				
		for(int i = 0; i < NUMCLASSES; ++i){
			for(int j = 0; j < NUMFEATURES; ++j){
				if(ancestors[j] != -1){
					if(j % (i+2) ==0){
						dependencies0[i][j] = 0.5 + generator.nextDouble() /2;
						dependencies1[i][j] = 0.5 - generator.nextDouble() /2;
					} else{
						dependencies0[i][j] = 0.5 - generator.nextDouble() /2;
						dependencies1[i][j] = 0.5 + generator.nextDouble() /2;
					}
				}
			}
		}
		
		//generate the data
		/*for(int i = 0; i < NUMCLASSES; ++i){
			for(int j = 0; j < SAMPLES_PER_CLASS; ++j){
				SamplePoint sam = new SamplePoint(i + 1);
				sam.feat[3] = (generator.nextDouble() < dependencies0[i][3]) ? 0:1;
				//generate 4 from 3
				if(sam.feat[3] == 0){
					sam.feat[4] = (generator.nextDouble() < dependencies0[i][4]) ? 0:1;
				} else {
					sam.feat[4] = (generator.nextDouble() < dependencies1[i][4]) ? 0:1;
				}
				
				//generate 7 from 3
				if(sam.feat[3] == 0){
					sam.feat[7] = (generator.nextDouble() < dependencies0[i][7]) ? 0:1;
				} else {
					sam.feat[7] = (generator.nextDouble() < dependencies1[i][7]) ? 0:1;
				}
				
				//generate 2 from 3
				if(sam.feat[3] == 0){
					sam.feat[2] = (generator.nextDouble() < dependencies0[i][2]) ? 0:1;
				} else {
					sam.feat[2] = (generator.nextDouble() < dependencies1[i][2]) ? 0:1;
				}
				
				//generate 5 from 4
				if(sam.feat[4] == 0){
					sam.feat[5] = (generator.nextDouble() < dependencies0[i][5]) ? 0:1;
				} else {
					sam.feat[5] = (generator.nextDouble() < dependencies1[i][5]) ? 0:1;
				}
				
				//generate 9 from 7
				if(sam.feat[7] == 0){
					sam.feat[9] = (generator.nextDouble() < dependencies0[i][9]) ? 0:1;
				} else {
					sam.feat[9] = (generator.nextDouble() < dependencies1[i][9]) ? 0:1;
				}
				
				//generate 8 from 9
				if(sam.feat[9] == 0){
					sam.feat[8] = (generator.nextDouble() < dependencies0[i][8]) ? 0:1;
				} else {
					sam.feat[8] = (generator.nextDouble() < dependencies1[i][8]) ? 0:1;
				}
				
				//generate 6 from 2
				if(sam.feat[2] == 0){
					sam.feat[6] = (generator.nextDouble() < dependencies0[i][6]) ? 0:1;
				} else {
					sam.feat[6] = (generator.nextDouble() < dependencies1[i][6]) ? 0:1;
				}
				
				//generate 10 from 2
				if(sam.feat[2] == 0){
					sam.feat[0] = (generator.nextDouble() < dependencies0[i][0]) ? 0:1;
				} else {
					sam.feat[0] = (generator.nextDouble() < dependencies1[i][0]) ? 0:1;
				}
				
				//generate 1 from 2
				if(sam.feat[2] == 0){
					sam.feat[1] = (generator.nextDouble() < dependencies0[i][1]) ? 0:1;
				} else {
					sam.feat[1] = (generator.nextDouble() < dependencies1[i][1]) ? 0:1;
				}
				
				data.add(sam);
			}

		}
		
		for(SamplePoint s: data){
			System.out.println(s);
		}*/
		
		
		
	}
	
	public static void main(String[] args){
		RNG r = new RNG();
		r.run();		
	}
	
	
}

