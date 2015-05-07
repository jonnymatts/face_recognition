package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum PersonAgeFine {
	BELOW26(0), BETWEEN26AND28(1), BETWEEN29AND31(2), BETWEEN32AND34(3), OVER34(4), UNKNOWN(-1);
	
	private int value;
	
	private PersonAgeFine(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
	
	public static PersonAgeFine valueOf(int age) {
		int id = -1;
		if((age < 26) && (age > 0)) id = 0;
		if((age > 25) && (age < 29)) id = 1;
		if((age > 28) && (age < 32)) id = 2;
		if((age > 31) && (age < 34)) id = 3;
		if(age > 34) id = 4;
		return map.get(id);
	}
	
	public static PersonAgeFine predictedValueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, PersonAgeFine> map = new HashMap<Integer, PersonAgeFine>();

    static {
        for (PersonAgeFine exEnum : PersonAgeFine.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
