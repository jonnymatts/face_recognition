package jonnymatts.facerecognition;

import static libsvm.svm.*;

import java.util.ArrayList;
import java.util.List;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class SVMHandler {
	
	private svm_model model;
	
	public PersonDataset predictClassesForBiometric(PersonDataset testingSet, Biometric biometric) {
		List<Person> pList = testingSet.getPersonList();
		for(Person p : pList) {
			int predictedClass = (int)svm_predict(model, getSVMNodeArray(p.getFeatureVector()));
			switch(biometric) {
				case AGE: p.predictedAge = PersonAge.valueOf(predictedClass); break;
				case ETHNICITY: p.predictedEthnicity = PersonEthnicity.valueOf(predictedClass); break;
				default: p.predictedGender = PersonGender.valueOf(predictedClass); break;
			}
		}
		testingSet.setPersonList(pList);
		return testingSet;
	}
	
	public void trainForBiometric(PersonDataset trainingSet, Biometric biometric) {
		model = svm_train(getSVMProblem(trainingSet, biometric), getSVMParameter());
	}
	
	public svm_node[] getSVMNodeArray(List<Double> featureVector) {
		List<svm_node> nList = new ArrayList<svm_node>();
		for(int j = 0; j < featureVector.size(); j++) {
			double val = featureVector.get(j);
			if(val != 0) {
				svm_node node = new svm_node();
				node.index = j;
				node.value = val;
				nList.add(node);
			}
		}
		svm_node node = new svm_node();
		node.index = -1;
		node.value = 0;
		nList.add(node);
		return nList.toArray(new svm_node[nList.size()]);
	}

	public svm_parameter getSVMParameter() {
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = 2;
		param.cache_size = 100;
		param.C = 100;
		param.gamma = 0.01;
		param.eps = 0.01;
		param.shrinking = 0;
		param.probability = 0;
		param.nr_weight = 0;
		return param;
	}
	
	public svm_problem getSVMProblem(PersonDataset set, Biometric biometric) {
		svm_problem prob = new svm_problem();
		prob.l = set.size();
		double[] y = new double[set.size()];
		prob.x = new svm_node[set.size()][];
		List<Person> pList = set.getPersonList();
		for(int i = 0; i < y.length; i++) {
			Person p = pList.get(i);
			switch(biometric) {
				case AGE: y[i] = p.age.getValue(); break;
				case ETHNICITY: y[i] = p.ethnicity.getValue(); break;
				default: y[i] = p.gender.getValue(); break;
			}
			List<Double> fv = p.getFeatureVector();
			svm_node[] nodeArray = getSVMNodeArray(fv);
			prob.x[i] = nodeArray;
		}
		prob.y = y;
		return prob;
	}
	
}
