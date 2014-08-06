package com;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.attributeSelection.LatentSemanticAnalysis;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

public class MultiLabelLogistic {

	public static void main(String args[]) throws Exception {

		/*
		 * 训练K个二分类器
		 */

		for (int k = 0; k < 28; k++) {

			File file = new File("src//file//svm//topic_only" + k + ".arff");

			ArffLoader loader = new ArffLoader();

			ArffSaver saver = new ArffSaver();

			loader.setFile(file);

			Instances ins = loader.getDataSet();
			ins.setClassIndex(ins.numAttributes() - 1);

			// ------------

			// 试试weka中 SVD(LSA, PCA)降维

			LatentSemanticAnalysis lsa = new LatentSemanticAnalysis();

			lsa.setMaximumAttributeNames(20);// 每个高层特征最多包含20个原始特征，线性表出

			lsa.setRank(20);// 降到20维

			lsa.buildEvaluator(ins);

			Instances ins_svd = lsa.transformedData(ins);
			

			// 写arff
			saver.setInstances(ins_svd);
			saver.setFile(new File("src//file//svm//topic_svd.arff"));
			saver.writeBatch();

			// ----------

			Logistic l = new Logistic();

			System.out.println("Start training logistic....." + k);

			l.buildClassifier(ins);

			System.out.println("build complete logistic" + k);

			Common.SaveModel(l, "logistic" + k + ".model");
		}

		// test on training set
		File file = new File("src//file//svm//topic_only0.arff");

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		Instances ins = loader.getDataSet();

		ins.setClassIndex(ins.numAttributes() - 1);

		int total_length = ins.numInstances();

		List<Logistic> logistic = new ArrayList<Logistic>();

		for (int k = 0; k < 28; k++) {
			Logistic l = (Logistic) Common
					.LoadModel("src//file//svm//model//logistic" + k + ".model");
			logistic.add(l);
		}

		List<Set<Integer>> predict_docs_labels = new ArrayList<Set<Integer>>();

		for (int i = 0; i < total_length; i++) {

			Instance testInst = ins.instance(i);

			System.out.println(i + ":");

			Set<Integer> predict_doc_labels = new HashSet<Integer>();

			for (int k = 0; k < 28; k++) {

				Logistic n = logistic.get(k);
				int predict_result = (int) n.classifyInstance(testInst);
				if (predict_result == 1)
					predict_doc_labels.add(k);
				System.out.print(predict_result + " ");
			}
			predict_docs_labels.add(predict_doc_labels);
			System.out.println();

		}

		/*
		 * F-measure
		 */

		MultiLabelClinicalCases mcc = Preprocess.getMultiLabelClinical();

		// Gold truth
		List<List<String>> docs_labels = mcc.multilabels;

		List<String> label_set = mcc.label_set;

		List<Set<Integer>> real_docs_labels = new ArrayList<Set<Integer>>();

		for (List<String> doc_labels : docs_labels) {

			Set<Integer> real_doc_labels = new HashSet<Integer>();

			for (String label : doc_labels) {

				real_doc_labels.add(label_set.indexOf(label));
			}
			real_docs_labels.add(real_doc_labels);

		}

		// Compare
		int precision_count = 0, total_prediction = 0, total_real = 0;
		for (int i = 0; i < total_length; i++) {

			Set<Integer> predict = predict_docs_labels.get(i);
			Set<Integer> real = real_docs_labels.get(i);

			total_prediction += predict.size();

			total_real += real.size();

			for (int e : predict) {
				if (real.contains(e))
					precision_count++;
			}

		}
		double precision = (double) precision_count / total_prediction;
		double recall = (double) precision_count / total_real;
		double f_measure = 2 * precision * recall / (precision + recall);
		System.out.println("precision : " + precision);
		System.out.println("recall : " + recall);
		System.out.println("f_measure : " + f_measure);

		Evaluator.logistic_five_fold(real_docs_labels);

	}

}
