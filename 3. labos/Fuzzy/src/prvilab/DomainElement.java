package prvilab;

import java.util.Arrays;

public class DomainElement {
	public int[] values;
	
	public DomainElement(int[] values) {
		this.values = values;
		
	}
	
	public int getNumberOfComponents(){
		return values.length;
		
	}
	
	public int getComponentValue(int index) {
		return values[index];
		
	}
	
	public static DomainElement of(int... elements) {
		return new DomainElement(elements);
	
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(values);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DomainElement other = (DomainElement) obj;
		if (!Arrays.equals(values, other.values))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return  Arrays.toString(values);
	}
	
	

}
