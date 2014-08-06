package com;

import java.io.File;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class LogisticForDB {

	public static void main(String args[]) throws Exception {

		Map<String, List<String>> map = getPreCom();

		/*
		 * 读取训练好的Logistic模型
		 */

		List<Logistic> logistic = new ArrayList<Logistic>();

		for (int k = 0; k < 28; k++) {
			Logistic l = (Logistic) Common
					.LoadModel("src//file//svm//model//logistic" + k + ".model");
			logistic.add(l);
		}

		/*
		 * 读训练集
		 */

		MultiLabelClinicalCases mcc = Preprocess.getMultiLabelClinical();

		List<String> label_set = mcc.label_set;

		List<String> vocab = mcc.vocab;

		// -----

		File file = new File("src//file//svm//topic0.arff");

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		Instances ins = loader.getDataSet();

		ins.setClassIndex(ins.numAttributes() - 1);

		Connection my = Preprocess.getConnectionMySql();

		System.out.println(vocab.size());

		int count = 0;

		for (String id : map.keySet()) {

			List<String> med_list = map.get(id);

			Instance instance = new Instance(vocab.size());

			if (!med_list.isEmpty()) {

				for (int i = 0, l = vocab.size(); i < l; i++) {

					if (med_list.contains(vocab.get(i)))
						instance.setValue(ins.attribute(i), 1);
					else
						instance.setValue(ins.attribute(i), 0);
				}
				instance.setDataset(ins);

				Set<Integer> predict_doc_labels = new HashSet<Integer>();

				for (int k = 0; k < 28; k++) {

					Logistic n = logistic.get(k);
					int predict_result = (int) n.classifyInstance(instance);
					if (predict_result == 1)
						predict_doc_labels.add(k);
					System.out.print(predict_result + " ");

				}
				System.out.println();

				String updateSql = "update prescription set function_class = ? where prescription_id= ?";

				StringBuilder sb = new StringBuilder();
				for (int e : predict_doc_labels) {
					sb.append(label_set.get(e) + " ");
				}

				PreparedStatement pstmt = my.prepareStatement(updateSql);

				pstmt.setString(1, sb.toString());

				pstmt.setString(2, id);

				pstmt.executeUpdate();

				my.commit();

				pstmt.close();

				count++;

				System.out.println(count);
			}
		}

		my.close();

	}

	public static Map<String, List<String>> getPreCom() throws SQLException {

		Connection my = Preprocess.getConnectionMySql();

		String sql = "select prescription_id as id, pre_dosage as dosage from prescription_simple_composition";

		PreparedStatement pstmt = my.prepareStatement(sql);
		ResultSet rs = pstmt.executeQuery();

		Map<String, List<String>> pre_com_map = new HashMap<String, List<String>>();

		while (rs.next()) {

			String id = rs.getString("id");

			String dosage = rs.getString("dosage").trim();

			String[] temp = dosage.split(" ");

			List<String> com_list = Arrays.asList(temp);

			pre_com_map.put(id, com_list);

		}

		rs.close();
		pstmt.close();
		my.close();

		return pre_com_map;
	}

}
