package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum PersonAge {
	BETWEEN10AND19(0), BETWEEN20AND29(1), BETWEEN30AND39(2), BETWEEN40AND49(3),
	BETWEEN50AND59(4), BETWEEN60AND69(5), OVER70(6), UNKNOWN(-1);
	
	private int value;
	
	private PersonAge(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
	
	public static PersonAge valueOf(int age) {
		int id = -1;
		if((age < 20) && (age > 0)) id = 0;
		if((age > 19) && (age < 30)) id = 1;
		if((age > 29) && (age < 40)) id = 2;
		if((age > 39) && (age < 50)) id = 3;
		if((age > 49) && (age < 60)) id = 4;
		if((age > 59) && (age < 70)) id = 5;
		if((age > 70)) id = 6;
		return map.get(id);
	}
	
	public static PersonAge predictedValueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, PersonAge> map = new HashMap<Integer, PersonAge>();

    static {
        for (PersonAge exEnum : PersonAge.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
