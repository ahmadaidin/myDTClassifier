package myclassifier;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.*;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.filters.*;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Instances;
import weka.core.converters.CSVLoader;


public class WekaAccessor {
	public WekaAccessor() {

	}

	public Instances loadArff(String filename) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		Instances data = new Instances(reader);
		reader.close();
		data.setClassIndex(data.numAttributes()-1);
		return data;
	}

	public Instances loadCSV(String filename) throws Exception {
		CSVLoader loader = new CSVLoader();
	    loader.setSource(new File(filename));
	    Instances data = loader.getDataSet();
	    
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public Instances removeAttributes(Instances instances, String rangeList, boolean invert) throws Exception {
		Instances inst;
		Instances instNew;
		Remove remove;

		inst = new Instances(instances);
		remove = new Remove();
		remove.setAttributeIndices(rangeList);
		remove.setInvertSelection(invert);
		remove.setInputFormat(inst);
		instNew = Filter.useFilter(inst, remove);
		return instNew;
	}

	public Instances resample(Instances data) {
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try {
			filter.setInputFormat(data);
			//filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(data, filter);
		} catch (Exception e) {
			//IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		return filteredIns;
	}

	/*public Id3 buildID3Classifier(Instances data) {
		Id3 tree = new Id3();         // new instance of tree
		tree.buildClassifier(data);
		return tree;
	}*/

	public J48 buildC45Classifier(Instances data) throws Exception {
		String[] options = new String[1];
		options[0] = "-U";            // unpruned tree
		J48 tree = new J48();         // new instance of tree
		tree.setOptions(options);     // set the options
		tree.buildClassifier(data);
		return tree;
	}

        public Id3 buildId3Classifier(Instances data) throws Exception {
		String[] options = new String[1];
		options[0] = "-U";            // unpruned tree
		Id3 tree = new Id3();         // new instance of tree
		tree.setOptions(options);     // set the options
		tree.buildClassifier(data);
		return tree;
	}
        
	public double[][] testModels(Classifier cModel, Instances trainingSet) throws Exception {

		Evaluation eTest = new Evaluation(trainingSet);
		eTest.evaluateModel(cModel, trainingSet);

		String strSummary = eTest.toSummaryString();
		System.out.println(strSummary);
	 
	 	// Get the confusion matrix
		double[][] cmMatrix = eTest.confusionMatrix();
		return cmMatrix;
	}

	public void testModel(Classifier cModel, Instances testData) {
		try {
			Evaluation eTest = new Evaluation(testData);
			eTest.evaluateModel(cModel, testData);
			
			System.out.println(eTest.toMatrixString());
			System.out.println(eTest.toClassDetailsString());
			System.out.println(eTest.toSummaryString());
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public void tenFoldCrossValidation(Classifier classifier, Instances dataset) {
		try {
			Evaluation eval = new Evaluation(dataset);
			eval.crossValidateModel(classifier, dataset, 10, new Random());
			
			System.out.println(eval.toMatrixString());
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toSummaryString());
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public ArrayList<Instances> percentageSplit(Instances data, int percent) {
		ArrayList<Instances> trainTest = new ArrayList<Instances>();
		int trainSize = (int) Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		trainTest.add(train);
		trainTest.add(test);
		train.randomize(new Random(0));

		return trainTest;
	}

	public void saveModel(Classifier c, String name, File path) throws Exception {
		ObjectOutputStream oos = null;
		try {
			oos = new ObjectOutputStream(
				new FileOutputStream(path + name + ".model"));
		
		} catch (FileNotFoundException e1) {
	    	e1.printStackTrace();
	    } catch (IOException e1) {
	    	e1.printStackTrace();
	    }
	    oos.writeObject(c);
	    oos.flush();
	    oos.close();
	}

	public Classifier loadModel(String name, File path) throws ClassNotFoundException, IOException {
		Classifier classifier;
		FileInputStream fis = new FileInputStream(path + name + ".model");
		ObjectInputStream ois = new ObjectInputStream(fis);
		classifier = (Classifier) ois.readObject();
		ois.close();

		return classifier;
	}

	public void classifyUnseenData(Classifier classifier, Instances unlabeled) throws Exception {
		for (int i = 0; i < unlabeled.numInstances(); i++) {
			double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
			System.out.println(clsLabel + " -> " + unlabeled.classAttribute().value((int) clsLabel));
			//labeled.instance(i).setClassValue(clsLabel);
		}
	}

}
