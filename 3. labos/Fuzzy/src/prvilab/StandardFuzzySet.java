package prvilab;

public class StandardFuzzySet{
	
	public StandardFuzzySet() {}
		
	public static IIntUnaryFunction lFunction(int alfa, int beta) {
			IIntUnaryFunction func = (index) -> {
				if (index < alfa) {
					return 1.0;
				}
				else if (index >= alfa && index < beta) {
					return (beta-index)/(double)(beta-alfa);
				}
				else if(index >= beta) {
					return 0.0;
				}
				else return 100.0;
			};
		return func;
	} 
	
	public static IIntUnaryFunction gammaFunction(int alfa, int beta) {
		IIntUnaryFunction func = (index) -> {
			if (index < alfa) {
				return 0;
			}
			else if (index >= alfa && index < beta) {
				return (index-alfa)/(double)(beta-alfa);
			}
			else if(index >= beta) {
				return 1;
			}
			else return 100.0;
		};
		return func;
	
	}
	
	public static IIntUnaryFunction lambdaFunction(int alfa, int beta, int gama) {
		IIntUnaryFunction func = (index) -> {
			if (index < alfa) {
				return 0;
			}
			else if (index >= alfa && index < beta) {
				return (index-alfa)/(double)(beta-alfa);
			}
			else if(index >= beta && index < gama) {
				return (gama-index)/(double)(gama-beta);
			}
			else if(index >= gama) {
				return 0;
			}
			else return 100.0;
		};
		return func;
		
	}	
	
}
	
	
	

