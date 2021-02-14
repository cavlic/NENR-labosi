package trecilab;

import prvilab.IFuzzySet;


public class Rule {
	
	private IFuzzySet[] antecedent;
	private IFuzzySet consequent;
	
	public Rule(IFuzzySet[] antecedent, IFuzzySet consequent) {
		this.antecedent = antecedent;
		this.consequent = consequent;
		
		
	};
	
	public IFuzzySet[] getAntecedent() {
		return this.antecedent;
	}
	
	public IFuzzySet getConsequent() {
		return this.consequent;
	}
	
	
	
}
