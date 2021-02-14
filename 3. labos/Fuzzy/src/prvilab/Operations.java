package prvilab;

import java.lang.Math;

public class Operations {
	
	public Operations() {}
	
	public static IFuzzySet unaryOperation(IFuzzySet set, IUnaryFunction uFun) {
		IDomain d = set.getDomain();
		MutableFuzzySet s = new MutableFuzzySet(d);
		for(DomainElement e : d) {
			s.set(e, uFun.valueAt(set.getValueAt(e)));
		}
		return (IFuzzySet) s;
		
	}
	
	public static IFuzzySet binaryOperation(IFuzzySet set1, IFuzzySet set2, IBinaryFunction bFun) {
		IDomain d = set1.getDomain();
		MutableFuzzySet s = new MutableFuzzySet(d);
		for(DomainElement e : d) {
			s.set(e, bFun.valueAt(set1.getValueAt(e), set2.getValueAt(e)));
		}
		return (IFuzzySet) s;
		
	}
	
	public static IUnaryFunction zadehNot() {
		IUnaryFunction uFun = (val) -> (1-val);
		return uFun;	
	}
	
	public static IBinaryFunction zadehAnd() {
		IBinaryFunction bFun = (val1, val2) -> Math.min(val1, val2);
		return bFun;
	}
	
	public static IBinaryFunction zadehOr() {
		IBinaryFunction bFun = (val1, val2) -> Math.max(val1, val2);
		return bFun;
	}
	
	public static IBinaryFunction hamacherTNorm(double par) {
		IBinaryFunction bFun = (val1, val2) -> (val1 * val2) / (double) (par + (1 - par) * (val1 + val2 - val1 * val2));
		return bFun;	
	}
	
	public static IBinaryFunction hamacherSNorm(double par) {
		IBinaryFunction bFun = (val1, val2) -> (val1 + val2 - (2 - par) * val1 * val2) / (double) (1 - (1 - par) * val1 * val2);
		return bFun;	
	}
	
	/* add other C-s or T or S norms */
	
	
	
}
