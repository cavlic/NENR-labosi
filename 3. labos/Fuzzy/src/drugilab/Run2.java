package drugilab;

import prvilab.Domain;
import prvilab.DomainElement;
import prvilab.IDomain;
import prvilab.IFuzzySet;
import prvilab.MutableFuzzySet;

public class Run2 {

	public static void main(String[] args) {
		IDomain u = Domain.intRange(1, 6); // {1,2,3,4,5}
		IDomain u2 = Domain.combine(u, u);
		IFuzzySet r1 = new MutableFuzzySet(u2)
					.set(DomainElement.of(1,1), 1)
					.set(DomainElement.of(2,2), 1)
					.set(DomainElement.of(3,3), 1)
					.set(DomainElement.of(4,4), 1)
					.set(DomainElement.of(5,5), 1)
					.set(DomainElement.of(3,1), 0.5)
					.set(DomainElement.of(1,3), 0.5);
				
		IFuzzySet r2 = new MutableFuzzySet(u2)
					.set(DomainElement.of(1,1), 1)
					.set(DomainElement.of(2,2), 1)
					.set(DomainElement.of(3,3), 1)
					.set(DomainElement.of(4,4), 1)
					.set(DomainElement.of(5,5), 1)
					.set(DomainElement.of(3,1), 0.5)
					.set(DomainElement.of(1,3), 0.1);
		
		IFuzzySet r3 = new MutableFuzzySet(u2)
					.set(DomainElement.of(1,1), 1)
					.set(DomainElement.of(2,2), 1)
					.set(DomainElement.of(3,3), 0.3)
					.set(DomainElement.of(4,4), 1)
					.set(DomainElement.of(5,5), 1)
					.set(DomainElement.of(1,2), 0.6)
					.set(DomainElement.of(2,1), 0.6)
					.set(DomainElement.of(2,3), 0.7)
					.set(DomainElement.of(3,2), 0.7)
					.set(DomainElement.of(3,1), 0.5)
					.set(DomainElement.of(1,3), 0.5);
		
		IFuzzySet r4 = new MutableFuzzySet(u2)
					.set(DomainElement.of(1,1), 1)
					.set(DomainElement.of(2,2), 1)
					.set(DomainElement.of(3,3), 1)
					.set(DomainElement.of(4,4), 1)
					.set(DomainElement.of(5,5), 1)
					.set(DomainElement.of(1,2), 0.4)
					.set(DomainElement.of(2,1), 0.4)
					.set(DomainElement.of(2,3), 0.5)
					.set(DomainElement.of(3,2), 0.5)
					.set(DomainElement.of(1,3), 0.4)
					.set(DomainElement.of(3,1), 0.4);
		
	
		System.out.println("r1 je definiran nad UxU? " + Relations.isUtimesUrelation(r1));
		
		System.out.println("r1 je refleksivna? " + Relations.isReflexive(r1));
		
		System.out.println("r3 je refleksivna? "+ Relations.isReflexive(r3));
		
		System.out.println("r1 je simetrična? " + Relations.isSymmetric(r1));
		
		System.out.println("r2 je simetrična? " + Relations.isSymmetric(r2));
		
		System.out.println("r3 je max-min tranzitivna? " + Relations.isMaxMinTransitive(r3));

		System.out.println("r4 je max-min tranzitivna? " + Relations.isMaxMinTransitive(r4));
		
		
	}

}

