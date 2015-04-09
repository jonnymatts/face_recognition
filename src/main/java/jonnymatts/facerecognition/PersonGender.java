package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum PersonGender {
	MALE(0), FEMALE(1);
	
	private int value;
	
	private PersonGender(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
	
	public static PersonGender valueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, PersonGender> map = new HashMap<Integer, PersonGender>();

    static {
        for (PersonGender exEnum : PersonGender.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
