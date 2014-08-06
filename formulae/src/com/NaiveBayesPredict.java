package com;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class NaiveBayesPredict {

	public static void main(String args[]) throws Exception {

		final String f = "src//file//med_doc.arff";

		Corpus c = Preprocess.getCorpus();

		int[] order = new int[c.med_doc.length];

		// for (int i = 0; i < order.length; i++)
		// order[i] = i;

		order = Common.ArrayRandomSort(c.med_doc.length);

		String[] med_doc_keys_inorder = c.map_keys;

		String[] med_doc_keys = new String[med_doc_keys_inorder.length];

		for (int d = 0, D = c.med_doc.length; d < D; d++) {
			med_doc_keys[d] = med_doc_keys_inorder[order[d]];
		}

		write_file(c, f, order);

		Instances ins = null;
		File file = new File(f);

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		ins = loader.getDataSet();
		ins.setClassIndex(ins.numAttributes() - 1);

		NaiveBayes n = new NaiveBayes();

		n.buildClassifier(ins);

		Evaluation testingEvaluation = new Evaluation(ins);

		int length = ins.numInstances();

		int correct = 0;

		for (int i = 0; i < length; i++) {

			Instance testInst = ins.instance(i);

			System.out.println(ins.classAttribute().value(
					(int) n.classifyInstance(testInst)));

			if (testInst.classValue() == n.classifyInstance(testInst))
				correct++;

			// if (label_set.contains((int) n.classifyInstance(testInst)))
			// correct++;
		}
		System.out.println((double) correct / length);

		System.out.println("分类器的正确率：" + (1 - testingEvaluation.errorRate()));

		System.out.println("F-Measure:" + testingEvaluation.fMeasure(24));

//		Evaluator.ten_fold(ins, c, order);
//
//		System.out.println(ins.numInstances());

	}

	public static void write_file(Corpus c, String filename, int[] doc_order)
			throws IOException {
		/*
		 * read medicine docs
		 */

		int[][] med_doc_inorder = c.med_doc;
		int[] labels_inorder = c.label_array;

		// 打乱顺序
		int[] order = doc_order;

		int[][] med_doc = new int[med_doc_inorder.length][];
		int[] labels = new int[labels_inorder.length];

		for (int d = 0, D = med_doc.length; d < D; d++) {

			int Nd = med_doc_inorder[order[d]].length;
			med_doc[d] = new int[Nd];
			for (int n = 0; n < Nd; n++) {
				med_doc[d][n] = med_doc_inorder[order[d]][n];
			}
			labels[d] = labels_inorder[order[d]];

		}

		List<String> label_set = c.label_set;

		int K = label_set.size();

		List<String> vocab = c.vocab;
		int V = vocab.size();

		/*
		 * write arff file for weka
		 */
		File file;
		OutputStream out;
		BufferedWriter bw;

		file = new File(filename);
		out = new FileOutputStream(file, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));

		bw.write("@relation medicine\n");
		bw.newLine();

		for (int v = 0; v < V; v++) {
			// bw.write("@attribute medicine" + v + " real\n");
			bw.write("@attribute medicine" + v + " real\n");
		}
		bw.write("@attribute function ");
		bw.write("{");

		StringBuilder s = new StringBuilder();
		for (int k = 0; k < K; k++)
			s.append(k + ", ");
		String str = s.substring(0, s.length() - 2);
		bw.write(str);
		bw.write("}\n");
		bw.newLine();

		bw.write("@data\n");

		for (int d = 0, D = med_doc.length; d < D; d++) {

			int[] tf = new int[V];
			for (int n = 0, Nd = med_doc[d].length; n < Nd; n++) {
				tf[med_doc[d][n]]++;
			}
			s = new StringBuilder();
			for (int e : tf) {
				s.append(e + ",");
			}

			s.append(labels[d]);
			bw.write(s + "\n");
		}

		bw.close();
		out.close();
		
		/*
		 * write file for LDA, BTM
		 */
		
		file = new File("src//file//docs_duplicate.txt");
		out = new FileOutputStream(file, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		
		for (int d = 0, D = med_doc.length; d < D; d++) {

			int Nd = med_doc[d].length;

			StringBuilder sb = new StringBuilder();
			
			for (int n = 0; n < Nd; n++) {
				sb.append(med_doc[d][n] + " ");
			}
			sb.append("," + labels[d]);
			bw.write(sb.toString() + "\n");

		}
		bw.close();
		out.close();
		
	}

	public NaiveBayesPredict(Corpus c, String filename, int[] doc_order)
			throws Exception {

		write_file(c, filename, doc_order);

		File file = new File(filename);

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		Instances ins = loader.getDataSet();
		ins.setClassIndex(ins.numAttributes() - 1);

		Evaluator.ten_fold(ins, c, doc_order);
	}
}
