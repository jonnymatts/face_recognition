package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum PersonEthnicity {
	NONINDIAN(0), INDIAN(1), UNKNOWN(-1);

	private int value;

	private PersonEthnicity(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
	
	public static PersonEthnicity valueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, PersonEthnicity> map = new HashMap<Integer, PersonEthnicity>();

    static {
        for (PersonEthnicity exEnum : PersonEthnicity.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
