package classifier;

//class to store the data
	public class ADSamplePoint{
		public int theClass;
		public int classifierGuess;
		public int[] features;
		
		public ADSamplePoint(int c, int w1, int w2, int w3, int w4, int w5, int w6, int w7, int w8, int w9, int w10){
			theClass = c;
			classifierGuess = 0;
			features = new int[10];
			features[1] = w1;
			features[2] = w2;
			features[3] = w3;
			features[4] = w4;
			features[5] = w5;
			features[6] = w6;
			features[7] = w7;
			features[8] = w8;
			features[9] = w9;
			features[0] = w10;		
		}
		
		public ADSamplePoint(int c){
			theClass = c;
			classifierGuess = 0;
			features = new int[10];
	
		}
		
		@Override
		public String toString(){
			return String.format("ADSamplePoint: Class - %d Guess - %d Features:[%d][%d][%d][%d][%d][%d][%d][%d][%d][%d] ", theClass, classifierGuess, features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[0]);
		}
		
	}