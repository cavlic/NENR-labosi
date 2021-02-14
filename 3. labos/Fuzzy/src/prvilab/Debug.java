package prvilab;

public class Debug {

	public static void print(IDomain domain, String headingText) {
		if(headingText!=null) {
			System.out.println(headingText);	
		}
		for (DomainElement e : domain) {
			System.out.println("Element domene: " + e);
		}
		System.out.println("Kardinalitet domene je: " + domain.getCardinality());
		System.out.println();
		
	}
	
	public static void print2(IFuzzySet set, String headingText) {
		if(headingText!=null) {
			System.err.println(headingText);	
		}
		IDomain domain = set.getDomain();
		for (DomainElement e : domain) {
			System.err.println("d(" + e + ")=" + set.getValueAt(e));
		}
		
		System.out.println();
		
	}
}
