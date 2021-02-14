package drugilab;

import prvilab.IDomain;
import prvilab.Domain;
import prvilab.DomainElement;
import prvilab.IFuzzySet;
import prvilab.MutableFuzzySet;


public class Relations {
	
	// checks if relation is defined under two equal universal sets
	public static boolean isUtimesUrelation(IFuzzySet relation) {
		Domain domena = (Domain) relation.getDomain();
		Domain u1 = (Domain) domena.getComponent(0);
		Domain u2 = (Domain) domena.getComponent(1);
		
		return u1.getDomena().equals(u2.getDomena());
		
	}
	
	public static boolean isReflexive(IFuzzySet relation) {
		if (isUtimesUrelation(relation)) {
			IDomain domena = relation.getDomain();
			IDomain u = domena.getComponent(0);
			int beginning = u.elementForIndex(0).values[0];
			int end = u.getCardinality();
			
			for(int i = beginning; i<=end; i++) {
				if(!(relation.getValueAt(DomainElement.of(i, i)) == 1)) {
					return false;
				}
			}
			return true;
		}
	
		else {
			return false;
		}
	}

	public static boolean isSymmetric(IFuzzySet relation) {
		if (isUtimesUrelation(relation)) {
			IDomain domena = relation.getDomain();
			IDomain u = domena.getComponent(0);
			int beginning = u.elementForIndex(0).values[0];
			int end = u.getCardinality();
			
			for(int i = beginning; i < end; i++) {
				for(int j = i+1; j <= end; j++) {
					if (!(relation.getValueAt(DomainElement.of(i,j)) == relation.getValueAt(DomainElement.of(j,i))))
					{
						return false;	
					}
				}
			}
			
			return true;
		}
		
		else {
			return false;
		}
	
		
	}
	
	public static boolean isMaxMinTransitive(IFuzzySet relation) {
		if (isUtimesUrelation(relation)) {
			IDomain domena = relation.getDomain();
			IDomain u = domena.getComponent(0);
			int beginning = u.elementForIndex(0).values[0];
			int end = u.elementForIndex(u.getCardinality()-1).values[0];
			
			IFuzzySet composition = compositionOfBinaryRelations(relation, relation);
					
			for (int i = beginning; i <= end; i++) {
				for (int j = beginning; j <= end; j++) {
					if (composition.getValueAt(DomainElement.of(i, j)) > relation.getValueAt(DomainElement.of(i, j))) {
						return false;
					}
				}
			}
			
			return true;
			
		}
		
		else {
			return false;
		}		

	}
	
	
	public static IFuzzySet compositionOfBinaryRelations(IFuzzySet set1, IFuzzySet set2) {
		IDomain domena1 = set1.getDomain();
		IDomain domena2 = set2.getDomain();
		
		IDomain X  = domena1.getComponent(0); // x domain
		Domain Y  = (Domain) domena1.getComponent(1); // y domain
		Domain Y_ = (Domain) domena2.getComponent(0); // y_ domain
		IDomain Z  = domena2.getComponent(1); // z domain
		
		if (! Y.getDomena().equals(Y_.getDomena())) {
			System.out.println("Nemoguće napraviti kompoziciju zbog dimenzija!");
		}
		
		MutableFuzzySet composition = new MutableFuzzySet(Domain.combine(X, Z));
		
		
		int z_beginning = Z.elementForIndex(0).values[0];
		int z_end = Z.elementForIndex(Z.getCardinality()-1).values[0];
		
		int x_beginning = X.elementForIndex(0).values[0];
		int x_end = X.elementForIndex(X.getCardinality()-1).values[0];
		
		int y_beginning = Y.elementForIndex(0).values[0];
		int y_end = Y.elementForIndex(Y.getCardinality()-1).values[0];
		
		for(int i = x_beginning; i <= x_end; i++) {
			for(int j = z_beginning; j <= z_end; j++) {
				double max = 0.0;
				
				for(int k = y_beginning; k <= y_end; k++) {
					double num = Math.min(set1.getValueAt(DomainElement.of(i, k)), set2.getValueAt(DomainElement.of(k, j)));
					if (num > max) {
						max = num;
					}
				}
				composition.set(DomainElement.of(i, j), max);
			}
			
		}
		
		return (IFuzzySet) composition;

	}
	
	public static boolean isFuzzyEquivalence(IFuzzySet r) {
		return (Relations.isReflexive(r) && Relations.isSymmetric(r) && Relations.isMaxMinTransitive(r));

	}
	
}
