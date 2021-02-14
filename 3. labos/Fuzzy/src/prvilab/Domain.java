package prvilab;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Domain implements IDomain{
	
	public static final int MAX_DIMENSION = 50;
	private List<DomainElement> domena = new ArrayList<DomainElement>();

	public Domain(List<DomainElement> domena) {
		this.domena = domena;	
	}
	
	public static IDomain intRange(int a, int b) {
		List<DomainElement> domena = new ArrayList<DomainElement>();
		while(a<b) {
			int[] intArray = new int[1];
			intArray[0] = a;
			domena.add(new DomainElement(intArray));
			a++;
		}
		return new Domain(domena);
		
	}
	
	
	public static Domain combine(IDomain first, IDomain second) {
		List<DomainElement> domena = new ArrayList<DomainElement>();
		for(DomainElement e : first) {
			for(DomainElement t : second) {
				int[] intArray = new int[MAX_DIMENSION];
				intArray = combine2(e.values, t.values);
				domena.add(new DomainElement(intArray));
			}
		}
		
		return new Domain(domena);
		
	}
	
	public static int[] combine2(int[] a, int[] b){
        int length = a.length + b.length;
        int[] result = new int[length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

  
	@Override
	public int getCardinality() {
		return domena.size();
	}

	@Override
	public IDomain getComponent(int numOfComponent) {
		List<DomainElement> retDomena = new ArrayList<DomainElement>();
		
		for(DomainElement e : domena) {
			int[] intArray = new int[1];
			intArray[0] = e.values[numOfComponent];
			DomainElement elem = new DomainElement(intArray);
			if (!retDomena.contains(elem)) {
				retDomena.add(elem);
			}
			
		}
		
		return new Domain(retDomena);
	}

	@Override
	public int getNumberOfComponents() {
		int num = 0;
		for(DomainElement e : domena) {
			num = e.getNumberOfComponents();
			break;
		}
		return num;
	}

	@Override
	public int indexOfElement(DomainElement element) {
		int index = 0;
		for(DomainElement e : domena) {
			if (e.equals(element)){
				return index;
			}
			else {
				index++;
			}
		}
		System.out.println("Element ne postoji u domeni");
		return -1;
		
	}

	@Override
	public DomainElement elementForIndex(int index) {
		DomainElement element = domena.get(index);
		return element;
	}
	
	public List<DomainElement> getDomena(){
		return this.domena;
	}
	
	@Override
	public Iterator<DomainElement> iterator() {
		return domena.iterator();
	}
	
}


