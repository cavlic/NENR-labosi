package trecilab;

import prvilab.DomainElement;
import prvilab.IFuzzySet;

public class COADefuzzifier implements IDefuzzify {

	@Override
	public int defuzzify(IFuzzySet set) {
		double mu = 0;
		double mu_times_element = 0;
		for (DomainElement e : set.getDomain()) {
			mu += set.getValueAt(e);
			mu_times_element += e.values[0] * set.getValueAt(e);
			
		}
		
		
		int result = (int) (mu_times_element / mu);
		System.err.println("Odluka: " + result);
		return result;
		
	}
	
	

}
