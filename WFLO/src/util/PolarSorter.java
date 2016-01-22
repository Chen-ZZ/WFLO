package util;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import my.weka.MyDenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class PolarSorter {

	public static void main(String[] args) throws Exception {

		DataSource source = new DataSource("arff/competition_3_polar_0.arff");
		Instances trainData = source.getDataSet();
		int numInstances = trainData.numInstances();

		if (trainData.classIndex() == -1) {
			trainData.setClassIndex(trainData.numAttributes() - 1);
		}

		double[] tempDistances = new double[(trainData.get(0).numAttributes() - 1) / 2];
		Map<Double, Double> tempAttributes = new HashMap<>();
		Instances tempData = new Instances(trainData);
		tempData.removeAll(tempData);

		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < (trainData.numAttributes() - 1) / 2; j++) {
				tempDistances[j] = trainData.get(i).value(2 * j);
				tempAttributes.put(trainData.get(i).value(2 * j), trainData.get(i).value(2 * j + 1));
			}

			Arrays.sort(tempDistances);

			List<Double> tempValues = new ArrayList<Double>();

			for (double distance : tempDistances) {
				tempValues.add(distance);
				tempValues.add(tempAttributes.get(distance));
			}
			tempValues.add(trainData.get(i).classValue());// the class value.

			Instance tempInstance = new MyDenseInstance(1.0,
					tempValues.stream().mapToDouble(Double::doubleValue).toArray());

			tempInstance.setDataset(trainData);
			tempData.add(tempInstance);
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(tempData);
		saver.setFile(new File("arff/test.arff"));
		saver.writeBatch();

	}
}
