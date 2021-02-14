package trecilab;


import prvilab.IFuzzySet;

public class AkcelFuzzySystemMin extends FuzzySystem {
	
	
	public AkcelFuzzySystemMin(IDefuzzify defuzzifier) {
		super(defuzzifier);
		
	}
	
	@Override
	public void initRules() {
		IFuzzySet[] r1_ant = new IFuzzySet[]{null, null, null, null, FuzzySets.SLOW, null};
       	IFuzzySet r1_cons = FuzzySets.SPEED_UP;
       	IFuzzySet[] r2_ant = new IFuzzySet[]{null, null, null, null, FuzzySets.FAST, null};
       	IFuzzySet r2_cons = FuzzySets.SLOW_DOWN;
        
       	addRule(new Rule(r1_ant, r1_cons));
       	addRule(new Rule(r2_ant, r2_cons));
    
		
	}

}
