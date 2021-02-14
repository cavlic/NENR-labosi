package prvilab;

public class CalculatedFuzzySet implements IFuzzySet{

	private IDomain d;
	private IIntUnaryFunction func;
	
	public CalculatedFuzzySet(IDomain d, IIntUnaryFunction func) {
		this.d = d;
		this.func = func;
		
	}
	
	@Override
	public IDomain getDomain() {
		return d;
	}

	@Override
	public double getValueAt(DomainElement e) {
		return func.valueAt(d.indexOfElement(e));	
	}

	@Override
	public IFuzzySet cutoff(double mu) {
		MutableFuzzySet s = new MutableFuzzySet(this.d);
	
		for (DomainElement e : this.d) {
			s = s.set(e, getValueAt(e) * mu);
		}
		
		return (IFuzzySet) s;
	}

}