package main;

import org.apache.log4j.FileAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;

public class mainES {

	// "ks1", "ks2", "competition_1", "competition_3"

	static String[] scenarios = { "competition_3" };

	static final Logger crossoverLogger = Logger.getLogger(mainES.class);
	static final Logger mutateLogger = Logger.getLogger(mainES.class);

	static PatternLayout layout = new PatternLayout("%m%n");

	static int mu = 5;
	static int lambda = 10;
	static int num_T = 100;
	static int maxEvaluations = 2000;
	static int runs = 30;

	public static void main(String argv[]) {

		for (String scenario : scenarios) {
			String crossoverLogFile = "Logs_(mu+lambda)/" + scenario + "_" + "(" + String.valueOf(mu) + "+"
					+ String.valueOf(lambda) + ")" + "_" + "ES_block_crossover.log";
			String mutateLogFile = "Logs_(mu+lambda)/" + scenario + "_" + "(" + String.valueOf(mu) + "+"
					+ String.valueOf(lambda) + ")" + "_" + "ES_block_mutate.log";

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

				// Crossover Operation
				crossoverLogger.removeAllAppenders();
				FileAppender crossoverAppender = new FileAppender(layout, crossoverLogFile, false);
				crossoverAppender.setImmediateFlush(true);
				crossoverLogger.addAppender(crossoverAppender);

				System.out.println(crossoverLogFile);

				for (int i = 0; i < runs; i++) {
					MuLambdaES es = new MuLambdaES(wfle);
					es.setMaxEvaluations(maxEvaluations);
					es.setNum_Turbines(num_T);
					es.setMu(mu);
					es.setLambda(lambda);
					es.setOperatorFlag("crossover");

					String bestResult = String.valueOf(es.run_Plus_ES());
					crossoverLogger.info(bestResult);
				}

				// Mutation Operation
				mutateLogger.removeAllAppenders();
				FileAppender mutateAppender = new FileAppender(layout, mutateLogFile, false);
				mutateAppender.setImmediateFlush(true);
				mutateLogger.addAppender(mutateAppender);

				System.out.println(mutateLogFile);

				for (int i = 0; i < runs; i++) {
					MuLambdaES es = new MuLambdaES(wfle);
					es.setMaxEvaluations(maxEvaluations);
					es.setNum_Turbines(num_T);
					es.setMu(mu);
					es.setLambda(lambda);
					es.setOperatorFlag("mutate");

					String bestResult = String.valueOf(es.run_Plus_ES());
					mutateLogger.info(bestResult);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}

		}

	}

}
