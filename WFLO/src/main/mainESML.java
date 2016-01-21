package main;

import org.apache.log4j.FileAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Utils;

public class mainESML {

	// "ks1", "ks2", "competition_1", "competition_3"

	static String[] scenarios = { "competition_3" };

	static final Logger crossoverLogger = Logger.getLogger(mainES.class);
	static final Logger mutateLogger = Logger.getLogger(mainES.class);

	static PatternLayout layout = new PatternLayout("%m%n");

	public static void main(String[] args) {

		int mu = 1;
		int lambda = 1;
		int num_T = 100;
		int maxEvaluations = 2000;

		int hiddenlayers = 20;

		String options_String = "-H " + hiddenlayers;

		for (String scenario : scenarios) {
			String crossover_ML_LogFile = "Logs_ML/" + scenario + "_" + String.valueOf(mu) + "_"
					+ String.valueOf(lambda) + "_" + "H-" + String.valueOf(hiddenlayers) + "_"
					+ "ES_block_crossover_ML.log";
			String mutate_ML_LogFile = "Logs_ML/" + scenario + "_" + String.valueOf(mu) + "_"
					+ String.valueOf(lambda) + "_" + "H-" + String.valueOf(hiddenlayers) + "_"
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

				crossoverLogger.removeAllAppenders();
				FileAppender crossoverAppender = new FileAppender(layout, crossover_ML_LogFile, false);
				crossoverAppender.setImmediateFlush(true);
				crossoverLogger.addAppender(crossoverAppender);

				System.out.println(crossover_ML_LogFile);

				for (int i = 0; i < 30; i++) {

					MuLambdaESML esml = new MuLambdaESML(wfle);
					esml.setMaxEvaluations(maxEvaluations);
					esml.setTrainEvaluations(maxEvaluations / 2);
					esml.setNum_Turbines(num_T);
					esml.setMu(mu);
					esml.setLambda(lambda);
					esml.setOperatorFlag("crossover");

					Classifier mlp = new MultilayerPerceptron();
					String[] options = Utils.splitOptions(options_String);
					((MultilayerPerceptron) mlp).setOptions(options);
					esml.setClassifier(mlp);

					String bestResult = String.valueOf(esml.run());
					crossoverLogger.info(bestResult);
				}

				mutateLogger.removeAllAppenders();
				FileAppender mutateAppender = new FileAppender(layout, mutate_ML_LogFile, false);
				mutateAppender.setImmediateFlush(true);
				mutateLogger.addAppender(mutateAppender);

				System.out.println(mutate_ML_LogFile);

				for (int i = 0; i < 30; i++) {

					MuLambdaESML esml = new MuLambdaESML(wfle);
					esml.setMaxEvaluations(maxEvaluations);
					esml.setTrainEvaluations(maxEvaluations / 2);
					esml.setNum_Turbines(num_T);
					esml.setMu(mu);
					esml.setLambda(lambda);
					esml.setOperatorFlag("mutate");

					Classifier mlp = new MultilayerPerceptron();
					String[] options = Utils.splitOptions(options_String);
					((MultilayerPerceptron) mlp).setOptions(options);
					esml.setClassifier(mlp);

					String bestResult = String.valueOf(esml.run());
					mutateLogger.info(bestResult);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}
}
