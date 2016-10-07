/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
//import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Instance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.ArrayList;
import java.util.Enumeration;


public class MyC45 extends Classifier
{
	/** The node's successors. */
	private MyC45[] m_Successors;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;
	private Discretize dFilter;
	private Instances train_data;
	/**
	 * Builds ID3 decision tree classifier.
	 *
	 * @param data the training data
	 * @exception Exception if classifier can't be built successfully
	 */
	public MyC45() {
		
	}
	
	public void buildClassifier(Instances data) throws Exception
	{
		/*
		//Need discretize filter to make bins. ID3Classifier will not work unless data is nominal
		dFilter = new Discretize();
		//apply discretize filter on data to create nominal data
		dFilter.setInputFormat(data);
		train_data = cloneInstances(data);
		train_data.delete();
		*/
		
            
		Instances train = data;
                Discretize filter = new Discretize();
                filter.setInputFormat(train);
                train = Filter.useFilter(data, filter);
		
                /*
		if (!train.classAttribute().isNominal())
		{
			throw new Exception("ID3: nominal class, please.");
		}*/
		
                /*
		Enumeration enumAtt = train.enumerateAttributes();

		while (enumAtt.hasMoreElements())
		{
			Attribute attr = (Attribute) enumAtt.nextElement();
			if (!attr.isNominal())
				throw new Exception("only nominal attributes, please.");

			Enumeration en = train.enumerateInstances();

			while (en.hasMoreElements())
			{
				if (((Instance) en.nextElement()).isMissing(attr))
					throw new Exception("no missing values, please.");
			}
		}
                */
		train = new Instances(train);
		train.deleteWithMissingClass();
		makeTree(train);
	}

	/**
	 * Method building ID3 tree.
	 *
	 * @param data the training data
	 * @exception Exception if decision tree can't be built successfully
	 */
	private void makeTree(Instances data) throws Exception {

	    // Check if no instances have reached this node.
	    if (data.numInstances() == 0) {
	      m_Attribute = null;
	      m_ClassValue = -1; //Instance.missingValue();
	      m_Distribution = new double[data.numClasses()];
	      return;
	    }

	    // Compute attribute with maximum information gain.
	    double[] gainRatios = new double[data.numAttributes()];
	    Enumeration attEnum = data.enumerateAttributes();
	    while (attEnum.hasMoreElements()) {
	      Attribute att = (Attribute) attEnum.nextElement();
	      gainRatios[att.index()] = computeGainRatio(data, att);
	    }
	    m_Attribute = data.attribute(Utils.maxIndex(gainRatios));
	    
	    // Make leaf if information gain is zero. 
	    // Otherwise create successors.

	    if (Utils.eq(gainRatios[m_Attribute.index()], 0)) {
	      m_Attribute = null;
	      m_Distribution = new double[data.numClasses()];
	      Enumeration instEnum = data.enumerateInstances();
	      while (instEnum.hasMoreElements()) {
	        Instance inst = (Instance) instEnum.nextElement();
	        m_Distribution[(int) inst.classValue()]++;
	      }
	      Utils.normalize(m_Distribution);
	      m_ClassValue = Utils.maxIndex(m_Distribution);
	      m_ClassAttribute = data.classAttribute();
	    } else {
	      Instances[] splitData = splitData(data, m_Attribute);
	      m_Successors = new MyC45[m_Attribute.numValues()];
	      for (int j = 0; j < m_Attribute.numValues(); j++) {
	        m_Successors[j] = new MyC45();
	        m_Successors[j].makeTree(splitData[j]);
	      }
	    }
	}
	/**
	 * Classifies a given test instance using the decision tree.
	 *
	 * @param instance the instance to be classified
	 * @return the classification
	 */
	public double classifyInstance(Instance instance)
	{
		Instance ins = instance;//discreting(instance);
		if (m_Attribute == null)
			return m_ClassValue;
		else
			return m_Successors[(int) ins.value(m_Attribute)].classifyInstance(ins);
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 *
	 * @param instance the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 */
	public double[] distributionForInstance(Instance instance) 
	    throws NoSupportForMissingValuesException {
		//instance = discreting(instance);
		if (m_Attribute == null) {
	      return m_Distribution;
	    } else { 
	      return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
	    }
	  }

	/**
	 * Prints the decision tree using the private toString method from below.
	 *
	 * @return a textual description of the classifier
	 */
	public String toString()
	{
		if ((m_Distribution == null) && (m_Successors == null))
			return "No model built yet.";

		return "\t==============Tree Diagram==============\n" + toString(0);
	}

	/**
	 * Computes information gain for an attribute.
	 *
	 * @param data the data for which info gain is to be computed
	 * @param att the attribute
	 * @return the information gain for the given attribute and data
	 */
	private double computeInfoGain(Instances data, Attribute att) throws Exception
	{
		double infoGain = computeEntropyFromData(data);
		Instances[] splitData = splitData(data, att);

		for (int j = 0; j < att.numValues(); j++)
		{
			if (splitData[j].numInstances() > 0)
			{
				infoGain -= ((double) splitData[j].numInstances() /
						(double) data.numInstances()) *
						computeEntropyFromData(splitData[j]);
			}
		}
		return infoGain;
	}

        
        private double computeSplitInfo(Instances data, Attribute att) throws Exception {
                double splitInfo = 0;                
		Instances[] splitData = splitData(data, att);

		for (int j = 0; j < att.numValues(); j++)
		{
			if (splitData[j].numInstances() > 0)
			{
                                splitInfo -= (double) (computeEntropy(splitData[j].numInstances()) / data.numInstances());
			}
		}
		return splitInfo;
            
        }
        
        private double computeGainRatio(Instances data, Attribute att) throws Exception {
            if (computeInfoGain(data, att) != 0 && computeSplitInfo(data, att) != 0)
                return computeInfoGain(data, att) / computeSplitInfo(data, att);            
            else
                return 0;
        }
        
	/**
	 * Computes the entropy of a dataset.
	 *
	 * @param data the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 */
        
        private double computeEntropy(int n) {
            return -n * Utils.log2(n);
        }
                
        
	private double computeEntropyFromData(Instances data) throws Exception
	{
		double [] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();

		while (instEnum.hasMoreElements())
		{
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}

		double entropy = 0;

		for (int j = 0; j < data.numClasses(); j++)
		{
			if (classCounts[j] > 0)                                
				entropy -= (double) (classCounts[j]/data.numInstances()) * Utils.log2((double) classCounts[j]/data.numInstances());
		}
		
		//return entropy + Utils.log2(data.numInstances());
                return entropy;
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 *
	 * @param data the data which is to be split
	 * @param att the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	private Instances[] splitData(Instances data, Attribute att)
	{
		Instances[] splitData = new Instances[att.numValues()];

		for (int j = 0; j < att.numValues(); j++)
			splitData[j] = new Instances(data, data.numInstances());

		Enumeration instEnum = data.enumerateInstances();

		while (instEnum.hasMoreElements())
		{
			Instance inst = (Instance) instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		return splitData;
	}

	/**
	 * Outputs a tree at a certain level.
	 *
	 * @param level the level at which the tree is to be printed
	 */
	private String toString(int level)
	{
		StringBuffer text = new StringBuffer();

		if (m_Attribute == null)
		{
			if (m_ClassValue == -1)
				text.append(": null");
			else
				text.append(": "+m_ClassAttribute.value((int) m_ClassValue));
		}
		else
		{
			for (int j = 0; j < m_Attribute.numValues(); j++)
			{
				text.append("\n");

				for (int i = 0; i < level; i++)
					text.append("|  ");

				text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
				text.append(m_Successors[j].toString(level + 1));
			}
		}
		return text.toString();
	}
	
	private Instance discreting(Instance instance) {		
		train_data.add(instance);
		try {
			train_data = Filter.useFilter(train_data, dFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		instance = train_data.instance(0);
		train_data.delete();
		return instance;
	}
	
	/*
	private Instances cloneInstances(Instances ins) {
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
		for(int i = 0; i < ins.numAttributes(); i++) {
			fvWekaAttributes.add(ins.attribute(i));
		}
		Instances instances = new Instances("copy_instance", fvWekaAttributes, 100);
		 //System.out.println(instances.toString());
		return instances;
	}
	*/
	

}
