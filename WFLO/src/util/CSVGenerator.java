package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.opencsv.CSVWriter;

public class CSVGenerator {

	static int runs = 30;

	private String[] readLines(String filename) throws IOException {
		FileReader fileReader = new FileReader(filename);

		BufferedReader bufferedReader = new BufferedReader(fileReader);
		List<String> lines = new ArrayList<String>();
		String line = null;

		while ((line = bufferedReader.readLine()) != null) {
			lines.add(line);
		}

		bufferedReader.close();

		return lines.toArray(new String[lines.size()]);
	}

	public static void main(String[] args) throws IOException {

		CSVGenerator generator = new CSVGenerator();
		CSVWriter csvWriter = new CSVWriter(new FileWriter("csv/results.csv"));

		File logFoler = new File("Logs_(mu+lambda)/");
		File[] listOfFiles = logFoler.listFiles();

		ArrayList<String> linesList = new ArrayList<String>();
		String[] linesArray = new String[listOfFiles.length];

		for (File file : listOfFiles) {
			linesList.add(file.getName());

		}

		csvWriter.writeNext(linesList.toArray(linesArray));
		linesList.removeAll(linesList);

		for (int i = 0; i < 30; i++) {
			for (int j = 0; j < listOfFiles.length; j++) {
				File file = listOfFiles[j];
				if (file.isFile() && file.getName().endsWith(".log")) {
					String[] temp = generator.readLines(file.getPath());
					if (i < temp.length) {
						linesList.add(temp[i]);
					}

				}
			}
			csvWriter.writeNext(linesList.toArray(linesArray));
			linesList.removeAll(linesList);
		}

		csvWriter.close();

	}

}
