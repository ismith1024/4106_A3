package classifier;

//class to store the data
	public class WineSamplePoint{
		int theClass;
		public int classifierGuess;
		public int[] features;
		
		public WineSamplePoint(int c, int w1, int w2, int w3, int w4, int w5, int w6, int w7, int w8, int w9, int w10, int w11, int w12, int w13){
			theClass = c;
			classifierGuess = 0;
			features = new int[WineClassifier.NUMFEATURES];
			features[1] = w1;
			features[2] = w2;
			features[3] = w3;
			features[4] = w4;
			features[5] = w5;
			features[6] = w6;
			features[7] = w7;
			features[8] = w8;
			features[9] = w9;
			features[10] = w10;
			features[11] = w11;
			features[12] = w12;
			features[0] = w13;
		}
		
	}
