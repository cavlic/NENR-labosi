package prvilab;

public class MutableFuzzySet implements IFuzzySet {

	private double[] memberships;
	private IDomain d;
	
	public MutableFuzzySet(IDomain d) {
		this.d = d;
		this.memberships = new double[d.getCardinality()];
	}
	
	@Override
	public IDomain getDomain(){
		return d;
	}

	@Override
	public double getValueAt(DomainElement e) {
		return memberships[d.indexOfElement(e)];
	}

	public MutableFuzzySet set(DomainElement e, double value) {
		int indexOfElem = d.indexOfElement(e);
		memberships[indexOfElem] = value;
		return this;
		
	}
	
	@Override
	public IFuzzySet cutoff(double mu) {
		for (int i = 0; i < memberships.length; i++) {
			memberships[i] = memberships[i] * mu;
		}
		
		return this;
	}

}


