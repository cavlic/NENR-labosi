package trecilab;


import prvilab.IFuzzySet;


public class KormiloFuzzySystemMin extends FuzzySystem {
	
	
	public KormiloFuzzySystemMin(IDefuzzify defuzzifier) {
		super(defuzzifier);
		
	}
	
	@Override
	public void initRules() {
        IFuzzySet[] r1_ant = new IFuzzySet[]{null, null, FuzzySets.REALLY_NEAR_SHORE, null, null, null};
        IFuzzySet r1_cons = FuzzySets.SHARP_TURN_RIGHT;
        IFuzzySet[] r2_ant = new IFuzzySet[]{null, null, null, FuzzySets.REALLY_NEAR_SHORE, null, null};
        IFuzzySet r2_cons = FuzzySets.SHARP_TURN_LEFT;
        IFuzzySet[] r3_ant = new IFuzzySet[]{null, null, FuzzySets.NEAR_SHORE, null, null, null};
        IFuzzySet r3_cons = FuzzySets.EASY_TURN_RIGHT;
        IFuzzySet[] r4_ant = new IFuzzySet[]{null, null, null, FuzzySets.NEAR_SHORE, null, null};
        IFuzzySet r4_cons = FuzzySets.EASY_TURN_LEFT;
        IFuzzySet[] r5_ant = new IFuzzySet[]{null, null, null, null, null, FuzzySets.WRONG_WAY};
        IFuzzySet r5_cons = FuzzySets.SHARP_TURN_LEFT;
        
        
        addRule(new Rule(r1_ant, r1_cons));
        addRule(new Rule(r2_ant, r2_cons));
        addRule(new Rule(r3_ant, r3_cons));
        addRule(new Rule(r4_ant, r4_cons));
        addRule(new Rule(r5_ant, r5_cons));
        


    

		
	}
	
}


