package trecilab;

import java.util.ArrayList;
import java.util.List;

import prvilab.DomainElement;
import prvilab.IFuzzySet;
import prvilab.Operations;

public class FuzzySystem {
	
	IDefuzzify defuzzifier;
	private List<Rule> rules = new ArrayList<Rule>();

	
	public FuzzySystem(IDefuzzify def) {
		this.defuzzifier = def;
				
	}
	
	public void addRule(Rule r) {
		rules.add(r);
	}
	
	public void initRules() {
		return;
	}
	

	public int conclude(int... values) {
		List<IFuzzySet> conclusions = new ArrayList<IFuzzySet>();
		
		for (Rule r : rules) {			
			for (int i = 0; i < r.getAntecedent().length; i++) {
				IFuzzySet antecedent = r.getAntecedent()[i];
				if (antecedent == null) {
					continue;
				}
				

				double mu = antecedent.getValueAt(DomainElement.of(values[i]));				
				conclusions.add(r.getConsequent().cutoff(mu));
									
				
			}

		}
		
		IFuzzySet result = conclusions.get(0);	

		for (int i = 1; i < conclusions.size(); i++) {
			result = Operations.binaryOperation(result, conclusions.get(i), Operations.zadehOr());
		}
		
		
		return defuzzifier.defuzzify(result);
			
	}


	
}
