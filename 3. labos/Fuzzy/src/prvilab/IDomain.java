package prvilab;

import java.lang.Iterable;

public interface IDomain extends Iterable<DomainElement> {
	int getCardinality();
	IDomain getComponent(int numOfComponent);
	int getNumberOfComponents();
	int indexOfElement(DomainElement element);
	DomainElement elementForIndex(int index);
	
}
