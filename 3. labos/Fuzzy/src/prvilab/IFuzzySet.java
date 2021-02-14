package prvilab;

public interface IFuzzySet {
	IDomain getDomain();
	double getValueAt(DomainElement e);
    IFuzzySet cutoff(double mu);

	
}

