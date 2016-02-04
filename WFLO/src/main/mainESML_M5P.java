package main;

import org.apache.log4j.FileAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;

import weka.classifiers.trees.M5P;
import weka.core.Utils;

public class mainESML_M5P {

	// "ks1", "ks2", "competition_1", "competition_3"

	static String[] scenarios = { "competition_1" };

	static final Logger crossoverLogger = Logger.getLogger(mainES.class);
	static final Logger mutateLogger = Logger.getLogger(mainES.class);

	static PatternLayout layout = new PatternLayout("%m%n");

	public static void main(String[] args) {

		int num_T = 100;

		int mu = 6;
		int lambda = 12;
		int runs = 30;
		int maxEvaluations = 2000;

		String EA_Stratergy = "Best"; // Pre-selection or Best-selection

		String dataFormat = "polar"; // Raw data format or Polar data format

		// Parameters for M5P Classifier
		int minNumInstances = 5; // Set minimum number of instances per leaf
		String options_M5P = "-M " + minNumInstances;

		M5P m5p = new M5P();

		try {
			m5p.setOptions(Utils.splitOptions(options_M5P));
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		for (String scenario : scenarios) {
			String crossover_ML_LogFile = "Logs_ML_" + EA_Stratergy + "/" + scenario + "_"
					+ String.valueOf(mu) + "_" + String.valueOf(lambda) + "_M5P_" + "M-"
					+ String.valueOf(minNumInstances) + "_" + "ES_block_crossover_ML.log";
			String mutate_ML_LogFile = "Logs_ML_" + EA_Stratergy + "/" + scenario + "_" + String.valueOf(mu)
					+ "_" + String.valueOf(lambda) + "_M5P_" + "M-" + String.valueOf(minNumInstances) + "_"
					+ "ES_block_mutate_ML.log";

			switch (scenario) {
			case "competition_1":
				num_T = 220;
				break;

			case "competition_3":
				num_T = 710;
				break;

			default:
				num_T = 100;
				break;
			}

			try {

				WindScenario ws = new WindScenario("Scenarios/" + scenario + ".xml");
				KusiakLayoutEvaluator wfle = new KusiakLayoutEvaluator();
				wfle.initialize(ws);

				// crossoverLogger.removeAllAppenders();
				// FileAppender crossoverAppender = new FileAppender(layout, crossover_ML_LogFile, false);
				// crossoverAppender.setImmediateFlush(true);
				// crossoverLogger.addAppender(crossoverAppender);
				//
				// System.out.println(crossover_ML_LogFile);
				//
				// for (int i = 0; i < runs; i++) {
				//
				// MuLambdaESML esml = null;
				//
				// if (EA_Stratergy.equals("Best")) {
				// esml = new MuLambdaESML_Best(wfle);
				// } else if (EA_Stratergy.equals("Pre")) {
				// esml = new MuLambdaESML_Pre(wfle);
				// }
				//
				// esml.setMaxEvaluations(maxEvaluations);
				// esml.setTrainEvaluations(maxEvaluations / 2);
				// esml.setNum_Turbines(num_T);
				// esml.setMu(mu);
				// esml.setLambda(lambda);
				// esml.setDataFormat(dataFormat);
				// esml.setOperatorFlag("crossover");
				// esml.setClassifier(m5p);
				//
				// String bestResult = String.valueOf(esml.run());
				// crossoverLogger.info(bestResult);
				// }

				mutateLogger.removeAllAppenders();
				FileAppender mutateAppender = new FileAppender(layout, mutate_ML_LogFile, false);
				mutateAppender.setImmediateFlush(true);
				mutateLogger.addAppender(mutateAppender);

				System.out.println(mutate_ML_LogFile);

				for (int i = 0; i < runs; i++) {

					MuLambdaESML esml = null;

					if (EA_Stratergy.equals("Best")) {
						esml = new MuLambdaESML_Best(wfle);
					} else if (EA_Stratergy.equals("Pre")) {
						esml = new MuLambdaESML_Pre(wfle);
					}

					esml.setMaxEvaluations(maxEvaluations);
					esml.setTrainEvaluations(maxEvaluations / 2);
					esml.setNum_Turbines(num_T);
					esml.setMu(mu);
					esml.setLambda(lambda);
					esml.setDataFormat(dataFormat);
					esml.setOperatorFlag("mutate");
					esml.setClassifier(m5p);

					String bestResult = String.valueOf(esml.run());
					mutateLogger.info(bestResult);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}
}
