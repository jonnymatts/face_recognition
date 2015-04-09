package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum PersonAge {
	BETWEEN10AND19(0), BETWEEN20AND29(0), BETWEEN30AND39(1), BETWEEN40AND49(2),
	BETWEEN50AND59(3), BETWEEN60AND69(4), OVER69(5);
	
	private int value;
	
	private PersonAge(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
	
	public static PersonAge valueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, PersonAge> map = new HashMap<Integer, PersonAge>();

    static {
        for (PersonAge exEnum : PersonAge.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
