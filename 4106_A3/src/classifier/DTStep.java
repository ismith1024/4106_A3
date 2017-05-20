package classifier;

import java.util.Arrays;
import java.util.Random;

public class DTStep {

	static final int VOID_DATA = -1;
	
	public int classIfTrue;
	public int classIfFalse;
	public DTStep dtStepIfTrue;
	public DTStep dtStepIfFalse;
	public DTStep parent;
	public int featureToTest;
	public boolean rand;
	public int[] rvs;
	
	public DTStep(int cit, int cif, int ftt, DTStep p){
		classIfTrue = VOID_DATA;
		classIfFalse = VOID_DATA;
		featureToTest = ftt;
		parent = p;
		rand = false;
	}
	
	public void eval(ADSamplePoint p){
		if(rand || featureToTest == -1 ||(classIfTrue == VOID_DATA && classIfFalse == VOID_DATA && dtStepIfTrue == null && dtStepIfFalse == null )){

			long seed = System.nanoTime();
			double ran = (new Random(seed).nextDouble());
			int ran2 = new Random(seed).nextInt(ArtificialDataClassifier.training.size());
			int tot = 0;
			for(int i = 1; i < rvs.length; ++i){
				tot += rvs[i];
			}
			int rClass = rvs.length -1;
			ran *= tot;
			double[] rvs2 = new double[rvs.length];
			int rtot = 0;
			for(int i = 1; i < rvs.length; ++i){
				rvs2[i] = (0.0 + rvs[i])/tot;
				rtot += rvs[i];
				if(ran < rtot) {
					rClass = i;
					break;
				} 
			}
System.out.println("Random class: " + rClass);				
			p.classifierGuess = ArtificialDataClassifier.training.get(ran2).theClass;

			return;
		}
		
		if(classIfTrue != VOID_DATA && p.features[featureToTest] == 1){
			p.classifierGuess = classIfTrue;
			return;
		}
		else if(classIfFalse != VOID_DATA && p.features[featureToTest] == 0){
			p.classifierGuess = classIfFalse;
			return;
		} else if(p.features[featureToTest] == 1 && dtStepIfTrue != null){
			dtStepIfTrue.eval(p);
		} else if(dtStepIfFalse != null) dtStepIfFalse.eval(p);
	}
	
	public void eval(WineSamplePoint p){
		if(classIfTrue != VOID_DATA && p.features[featureToTest] == 1){
			p.classifierGuess = classIfTrue;
			return;
		}
		else if(classIfFalse != VOID_DATA && p.features[featureToTest] == 0){
			p.classifierGuess = classIfFalse;
			return;
		} else if(featureToTest != VOID_DATA && p.features[featureToTest] == 1 && dtStepIfTrue != null){
			dtStepIfTrue.eval(p);
		} else if(featureToTest != VOID_DATA && dtStepIfFalse != null) dtStepIfFalse.eval(p);
		else{
			long seed = System.nanoTime();
			double ran = (new Random(seed).nextDouble());
			int ran2 = new Random(seed).nextInt(WineClassifier.training.size());
			p.classifierGuess = WineClassifier.training.get(ran2).theClass;
		}
	}
	
	@Override
	public String toString(){
		if(rand) return "Arbitrary";
		
		String vip = "null";
		String vin = "null";
		if(dtStepIfTrue != null) vip = "" + dtStepIfTrue.featureToTest;
		if(dtStepIfFalse != null) vin = "" + dtStepIfFalse.featureToTest;
				
		return(" Feature: " + featureToTest + " .. 1 -> Classify " + classIfTrue + " .. 0 -> Classify " + classIfFalse);
	}
	
	
}
