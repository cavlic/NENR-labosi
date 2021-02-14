package prvilab;

public class Run {
	
	public static void main(String[] args) {
		IDomain d1 = Domain.intRange(0, 5);
		Debug.print(d1, "Elementi domene d1:");
		
		IDomain d2 = Domain.intRange(0, 3);
		Debug.print(d2, "Elementi domene d2:");
		
		IDomain d3 = Domain.combine(d1, d2);
		Debug.print(d3, "Elementi domene d3:");
		
		System.out.println(d3.elementForIndex(0));
		System.out.println(d3.elementForIndex(5));
		System.out.println(d3.elementForIndex(14));
		
		System.out.println();
		System.out.println(d3.indexOfElement(DomainElement.of(4,1)));
		System.out.println(d3.getNumberOfComponents());
		System.out.println();
	
		
		IDomain d4 = d3.getComponent(0);
		Debug.print(d4, "Vraćena domena d1:");
		
		
		IDomain d5 = Domain.intRange(0, 11);
		IFuzzySet set1 = new MutableFuzzySet(d5)
								.set(d5.elementForIndex(0), 1.0)
								.set(d5.elementForIndex(1), 0.8)
								.set(DomainElement.of(2), 0.6)
								.set(DomainElement.of(3), 0.4)
								.set(DomainElement.of(4), 0.2);
		
		Debug.print2(set1, "MutableFuzzySet: ");
		
		IDomain d6 = Domain.intRange(-5, 6);
		IFuzzySet set2 = new CalculatedFuzzySet(d6, StandardFuzzySet.lambdaFunction(
												d6.indexOfElement(DomainElement.of(-4)),
											    d6.indexOfElement(DomainElement.of(0)),
											    d6.indexOfElement(DomainElement.of(4))
											    )
		);
		
		IFuzzySet set3 = new CalculatedFuzzySet(d6, StandardFuzzySet.lFunction(
			    d6.indexOfElement(DomainElement.of(0)),
			    d6.indexOfElement(DomainElement.of(4))
			    )
		);
		IFuzzySet set4 = new CalculatedFuzzySet(d6, StandardFuzzySet.gammaFunction(
			    d6.indexOfElement(DomainElement.of(0)),
			    d6.indexOfElement(DomainElement.of(4))
			    )
		);
		Debug.print2(set2, "CalculatedFuzzySet: lambdaFunction ");
		Debug.print2(set3, "CalculatedFuzzySet: lFunction");
		Debug.print2(set4, "CalculatedFuzzySet: gammaFunction");
		
		IFuzzySet notSet1 = Operations.unaryOperation(set1, Operations.zadehNot());
		Debug.print2(notSet1, "notSet1:");
		
		IFuzzySet union = Operations.binaryOperation(set1, notSet1, Operations.zadehOr());
		Debug.print2(union, "Set1 union Set2");
		
		IFuzzySet hinters = Operations.binaryOperation(set1, notSet1, Operations.hamacherTNorm(1.0));
		Debug.print2(hinters, "Set1 intersection with notSet1 using parameterised Hamacher T norm with parameter 1.0:");
			
	}
}
