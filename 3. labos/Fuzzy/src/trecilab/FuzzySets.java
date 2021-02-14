package trecilab;

import prvilab.CalculatedFuzzySet;
import prvilab.Domain;
import prvilab.DomainElement;
import prvilab.IDomain;
import prvilab.IFuzzySet;
import prvilab.StandardFuzzySet;

public class FuzzySets {
	
	
	// DOMENE
	
	public static final IDomain DISTANCE = Domain.intRange(0, 1301);
	public static final IDomain ANGLE = Domain.intRange(-90, 91);
	public static final IDomain VELOCITY = Domain.intRange(0, 101);
	public static final IDomain ACCELERATION = Domain.intRange(-50, 51);
	
	
	// AKO //
	
	public static final IFuzzySet REALLY_NEAR_SHORE = new CalculatedFuzzySet(DISTANCE,
			StandardFuzzySet.lFunction(DISTANCE.indexOfElement(DomainElement.of(40)),
			DISTANCE.indexOfElement(DomainElement.of(50))));	
	
	public static final IFuzzySet NEAR_SHORE = new CalculatedFuzzySet(DISTANCE,
			StandardFuzzySet.lFunction(DISTANCE.indexOfElement(DomainElement.of(50)),
			DISTANCE.indexOfElement(DomainElement.of(60))));

	
	public static final IFuzzySet REALLY_FAR_FROM_SHORE = new CalculatedFuzzySet(DISTANCE,
			StandardFuzzySet.gammaFunction(DISTANCE.indexOfElement(DomainElement.of(350)),
			DISTANCE.indexOfElement(DomainElement.of(500))));
	
    public static final IFuzzySet WRONG_WAY = new CalculatedFuzzySet(DISTANCE,
    		StandardFuzzySet.lFunction(DISTANCE.indexOfElement(DomainElement.of(0)),
    		DISTANCE.indexOfElement(DomainElement.of(1))));
    
    public static final IFuzzySet SLOW = new CalculatedFuzzySet(VELOCITY,
    		StandardFuzzySet.lFunction(VELOCITY.indexOfElement(DomainElement.of(40)),
    		VELOCITY.indexOfElement(DomainElement.of(60))));
    
    public static final IFuzzySet FAST = new CalculatedFuzzySet(VELOCITY,
    		StandardFuzzySet.gammaFunction(VELOCITY.indexOfElement(DomainElement.of(60)),
    		VELOCITY.indexOfElement(DomainElement.of(70))));
    
    
    // ONDA //
	
	public static final IFuzzySet SHARP_TURN_RIGHT = new CalculatedFuzzySet(ANGLE,
			StandardFuzzySet.lFunction(ANGLE.indexOfElement(DomainElement.of(-90)), 
			ANGLE.indexOfElement(DomainElement.of(-70))));
	
	public static final IFuzzySet SHARP_TURN_LEFT = new CalculatedFuzzySet(ANGLE,
			StandardFuzzySet.gammaFunction(ANGLE.indexOfElement(DomainElement.of(70)),
			ANGLE.indexOfElement(DomainElement.of(90))));
	
	public static final IFuzzySet EASY_TURN_RIGHT = new CalculatedFuzzySet(ANGLE,
			StandardFuzzySet.lFunction(ANGLE.indexOfElement(DomainElement.of(-70)),
			ANGLE.indexOfElement(DomainElement.of(-60))));
	
	public static final IFuzzySet EASY_TURN_LEFT = new CalculatedFuzzySet(ANGLE, 
			StandardFuzzySet.gammaFunction(ANGLE.indexOfElement(DomainElement.of(60)),
			ANGLE.indexOfElement(DomainElement.of(70))));
	
	public static final IFuzzySet SLOW_DOWN = new CalculatedFuzzySet(ACCELERATION, 
			StandardFuzzySet.lFunction(ACCELERATION.indexOfElement(DomainElement.of(-25)),
			ACCELERATION.indexOfElement(DomainElement.of(0))));
	
	public static final IFuzzySet SPEED_UP = new CalculatedFuzzySet(ACCELERATION, 
			StandardFuzzySet.gammaFunction(ACCELERATION.indexOfElement(DomainElement.of(0)),
			ACCELERATION.indexOfElement(DomainElement.of(25))));
	



	
	

}
