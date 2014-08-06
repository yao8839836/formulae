package com;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.sql.Connection;
import java.sql.DriverManager;
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

public class Preprocess {
	static File result = null;
	static OutputStream out = null;
	static BufferedWriter bw = null;

	public static Corpus getCorpus() throws IOException {

		/*
		 * 生成文件列表
		 */
		String path = "E:\\pre_enrich";
		result = new File("src//file//doclist(Pre_enrich).txt");
		out = new FileOutputStream(result, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		readfile(path);
		bw.close();
		out.close();

		/*
		 * 读取方剂组成、类别
		 */
		Map<String, Set<Integer>> label_map = new HashMap<String, Set<Integer>>();// 构建文档-label
		// set
		// 键值对
		String[] map_keys = new String[3428];

		List<String> distinct_map_keys = new ArrayList<String>();

		String[] texts = new String[3428];
		List<String> vocab = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<String[]> formulae = new ArrayList<String[]>();

		String[] function = new String[3428];

		File f = new File("src//file//doclist(Pre_enrich).txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";
		int file_count = 0;
		while ((line = reader.readLine()) != null) {

			if (!line.contains(",")) {

				String pre_name = line.substring(line.lastIndexOf("\\") + 1,
						line.indexOf("."));

				f = new File(line);
				BufferedReader reader_1 = new BufferedReader(
						new InputStreamReader(new FileInputStream(f), "UTF-8"));
				String text = "";
				while ((line = reader_1.readLine()) != null) {
					text += line;

				}

				if (text.contains("由") && text.contains("组成")) {

					int begin = text.indexOf("由");
					int end = text.indexOf("组成");
					String composition = text.substring(begin + 1, end);
					texts[file_count] = text.substring(0, begin)
							+ text.substring(end, text.length() - 1);
					if (composition.contains("等")) {
						System.out.println(f + "  " + composition);
						composition = composition.substring(0, composition
								.indexOf("等"));
					}

					System.out.println(f + "  " + composition);
					String category = "";
					String s = f.toString();
					category = s.substring(14, s.lastIndexOf("\\"));// 14,8
					System.out.println(category);
					function[file_count] = category;

					String[] medicines = composition.split("、");

					StringBuilder sb = new StringBuilder(pre_name + ",");

					for (int i = 0; i < medicines.length; i++) {
						if (medicines[i].contains("（")) {
							medicines[i] = medicines[i].substring(0,
									medicines[i].indexOf("（"));
						}

						sb.append(medicines[i] + " ");
					}

					map_keys[file_count] = sb.toString();

					formulae.add(medicines);

					if (!distinct_map_keys.contains(map_keys[file_count])) {

						distinct_map_keys.add(map_keys[file_count]);
					}

					for (int i = 0; i < medicines.length; i++) {
						// System.out.print(medicines[i] + " ");
						if (!vocab.contains(medicines[i]))
							vocab.add(medicines[i]);
					}

					System.out.println();

				}
				reader_1.close();
			} else {

				map_keys[file_count] = line.substring(
						line.lastIndexOf("\\") + 1, line.indexOf(".")).trim();

				String str = line.substring(line.indexOf(",") + 1,
						line.length()).trim();

				String[] medicines = str.split(" ");

				String category = line.substring(14, line.lastIndexOf("\\"));// 14,8

				function[file_count] = category;

				formulae.add(medicines);

				if (!distinct_map_keys.contains(map_keys[file_count])) {

					distinct_map_keys.add(map_keys[file_count]);
				}

				for (String e : medicines) {

					if (!vocab.contains(e))
						vocab.add(e);

				}

			}

			file_count++;

		}
		reader.close();

		// System.out.println(file_count);
		// System.out.println(vocab.size());

		for (int i = 0; i < function.length; i++) {
			System.out.println(function[i]);
			if (!labels.contains(function[i])) {
				labels.add(function[i]);
			}
		}
		/*
		 * 构建文档
		 */
		int[][] med_doc = new int[3428][];
		int[] label_array = new int[med_doc.length];
		for (int i = 0; i < med_doc.length; i++) {
			med_doc[i] = new int[formulae.get(i).length];
			String[] medicines = formulae.get(i);
			for (int j = 0; j < med_doc[i].length; j++) {
				int index = vocab.indexOf(medicines[j]);
				med_doc[i][j] = index;
				System.out.print(index + " ");
			}
			System.out.println("label: " + labels.indexOf(function[i]));
			label_array[i] = labels.indexOf(function[i]);

		}
		/*
		 * 方剂功效Map
		 */

		for (int i = 0; i < map_keys.length; i++) {

			String key = map_keys[i];
			Set<Integer> label_set = new HashSet<Integer>();

			for (int j = 0; j < function.length; j++) {

				if (key.equals(map_keys[j])) {
					label_set.add(labels.indexOf(function[j]));
				}
			}
			label_map.put(key, label_set);

		}

		for (String e : label_map.keySet()) {

			System.out.println(label_map.get(e).toString());
		}

		/*
		 * 构建去重文档集
		 */
		int[][] distinct_med_doc = new int[distinct_map_keys.size()][];

		int[] distinct_label_array = new int[distinct_map_keys.size()];

		for (int i = 0, l = distinct_map_keys.size(); i < l; i++) {

			String e = distinct_map_keys.get(i);
			String[] medicines = e.substring(e.indexOf(",") + 1, e.length())
					.split(" ");
			distinct_med_doc[i] = new int[medicines.length];

			for (int j = 0; j < distinct_med_doc[i].length; j++) {

				int index = vocab.indexOf(medicines[j]);
				distinct_med_doc[i][j] = index;
				System.out.print(index + " ");
			}
			Set<Integer> label = label_map.get(e);

			for (int first : label) {

				distinct_label_array[i] = first;
				break;
			}

		}

		Corpus corpus = new Corpus(med_doc, label_array, labels, vocab,
				map_keys, label_map, distinct_med_doc, distinct_map_keys,
				distinct_label_array);

		return corpus;
	}

	public static MedLabels getZLabels() throws IOException {

		/*
		 * 生成文件列表
		 */

		String path = "E:\\yao_enrich";
		result = new File("src//file//doclist(Med_enrich).txt");
		out = new FileOutputStream(result, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		readfile(path);
		bw.close();
		out.close();

		File f = new File("src//file//doclist(Med_enrich).txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		String[] medicines = new String[97302];
		String[] med_classes = new String[97302];

		int line_count = 0;
		while ((line = reader.readLine()) != null) {

			String label = line.substring(14, line.lastIndexOf("\\"));// 7,14
			String medicine = line.substring(line.lastIndexOf("\\") + 1, line
					.lastIndexOf("."));

			// System.out.println(medicine + " " +label);

			medicines[line_count] = medicine;
			med_classes[line_count] = label;

			line_count++;

		}

		reader.close();

		MedLabels labels = new MedLabels(medicines, med_classes);

		return labels;
	}

	public static boolean readfile(String filepath)
			throws FileNotFoundException, IOException {

		try {

			File file = new File(filepath);
			if (!file.isDirectory()) {
				System.out.println("文件");
				System.out.println("path=" + file.getPath());
				System.out.println("absolutepath=" + file.getAbsolutePath());
				System.out.println("name=" + file.getName());

			} else if (file.isDirectory()) {
				System.out.println("文件夹");
				String[] filelist = file.list();
				for (int i = 0; i < filelist.length; i++) {
					File readfile = new File(filepath + "\\" + filelist[i]);
					if (!readfile.isDirectory()) {
						System.out.println("path=" + readfile.getPath());
						System.out.println("absolutepath="
								+ readfile.getAbsolutePath());
						bw.write(readfile.getAbsolutePath());
						bw.newLine();
						System.out.println("name=" + readfile.getName());

					} else if (readfile.isDirectory()) {
						readfile(filepath + "\\" + filelist[i]);
					}
				}

			}

		} catch (FileNotFoundException e) {
			System.out.println("readfile()   Exception:" + e.getMessage());
		}
		return true;
	}

	public static List<String> getMustCannotLink() throws SQLException {

		Connection my = getConnectionMySql();

		String sql = "select * from med_relation";

		PreparedStatement pstmt = my.prepareStatement(sql);
		ResultSet rs = pstmt.executeQuery();

		List<String> constraints = new ArrayList<String>();

		while (rs.next()) {

			String med_name_a = rs.getString("med_name_a");
			String med_name_b = rs.getString("med_name_b");

			String relation = rs.getString("med_relation");

			constraints.add(med_name_a + " " + med_name_b + " " + relation);
		}

		rs.close();
		pstmt.close();
		my.close();

		return constraints;
	}

	public static Connection getConnectionMySql() {
		String url = "jdbc:mysql://10.15.62.29:3306/tcm?useUnicode=true&characterEncoding=utf8";
		String user = "root";
		String psw = "123";
		Connection conn = null;
		try {
			Class.forName("com.mysql.jdbc.Driver");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		try {
			conn = DriverManager.getConnection(url, user, psw);
			conn.setAutoCommit(false);
			return conn;
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static ClinicalCases getClinical() throws IOException {

		/*
		 * 生成文件列表
		 */
		String path = "E:\\clinical-class";
		result = new File("src//file//doclist(clinic).txt");
		out = new FileOutputStream(result, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		readfile(path);
		bw.close();
		out.close();

		File f = new File("src//file//doclist(clinic).txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		List<String> vocab = getMedicines();
		List<List<Integer>> med_doc_list = new ArrayList<List<Integer>>();
		List<String> doc_label = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {

			File file = new File(line);

			BufferedReader reader_1 = new BufferedReader(new InputStreamReader(
					new FileInputStream(file), "UTF-8"));

			doc_label.add(line.substring(line.lastIndexOf("s") + 2, line
					.lastIndexOf("\\")));

			line = "";

			line = reader_1.readLine();

			line = reader_1.readLine();

			reader_1.close();

			/*
			 * 构建文档集
			 */

			String[] medicines = line.split(" ");

			List<Integer> formula = new ArrayList<Integer>();

			for (String medicine : medicines) {
				if (vocab.contains(medicine)) {

					formula.add(vocab.indexOf(medicine));
					System.out.println(vocab.indexOf(medicine) + "　");
				}

			}
			System.out.println();
			med_doc_list.add(formula);

		}

		reader.close();

		int[][] med_doc = new int[med_doc_list.size()][];

		for (int i = 0; i < med_doc.length; i++) {

			List<Integer> formula = med_doc_list.get(i);
			med_doc[i] = new int[formula.size()];

			for (int j = 0; j < med_doc[i].length; j++)
				med_doc[i][j] = formula.get(j);
		}

		/*
		 * 文档单个label
		 */

		String[] labels = new String[doc_label.size()];

		for (int i = 0; i < labels.length; i++)
			labels[i] = doc_label.get(i);

		ClinicalCases cc = new ClinicalCases(med_doc, labels);

		return cc;

	}

	public static MultiLabelClinicalCases getMultiLabelClinical()
			throws IOException {

		/*
		 * 生成文件列表
		 */
		String path = "E:\\clinic-multilabel";
		result = new File("src//file//doclist(multilabel-clinic).txt");
		out = new FileOutputStream(result, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		readfile(path);
		bw.close();
		out.close();

		File f = new File("src//file//doclist(multilabel-clinic).txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		List<String> vocab = getMedicines();
		List<List<Integer>> med_doc_list = new ArrayList<List<Integer>>();

		List<List<String>> doc_multilabel = new ArrayList<List<String>>();

		List<String> label_set = new ArrayList<String>();

		List<String> new_vocab = new ArrayList<String>();

		// 得到医案中的词表（单味药）
		while ((line = reader.readLine()) != null) {

			File file = new File(line);

			BufferedReader reader_1 = new BufferedReader(new InputStreamReader(
					new FileInputStream(file), "UTF-8"));

			reader_1.readLine();
			String line_2 = reader_1.readLine().trim();
			reader_1.close();

			String[] medicines = line_2.split(" ");

			for (String medicine : medicines) {
				if (!new_vocab.contains(medicine) && vocab.contains(medicine)) {
					new_vocab.add(medicine);
				}
			}

		}

		reader.close();

		// 再读一遍，用新的词表构建文档集
		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));

		while ((line = reader.readLine()) != null) {

			File file = new File(line);

			BufferedReader reader_1 = new BufferedReader(new InputStreamReader(
					new FileInputStream(file), "UTF-8"));

			String line_1 = reader_1.readLine().trim();
			String line_2 = reader_1.readLine().trim();

			reader_1.close();

			/*
			 * 文档多个标签
			 */
			String[] labels = line_1.split(" ");
			// 标签集合，大小等于主题数
			for (String label : labels) {
				if (!label_set.contains(label)) {
					label_set.add(label);
				}
			}
			List<String> doc_labels = Arrays.asList(labels);

			doc_multilabel.add(doc_labels);

			/*
			 * 构建文档集
			 */
			String[] medicines = line_2.split(" ");

			List<Integer> formula = new ArrayList<Integer>();

			for (String medicine : medicines) {
				if (new_vocab.contains(medicine)) {

					formula.add(new_vocab.indexOf(medicine));
					System.out.println(new_vocab.indexOf(medicine) + "　");
				}

			}
			System.out.println();
			med_doc_list.add(formula);

		}
		reader.close();

		int[][] med_doc = new int[med_doc_list.size()][];

		for (int i = 0; i < med_doc.length; i++) {

			List<Integer> formula = med_doc_list.get(i);
			med_doc[i] = new int[formula.size()];

			for (int j = 0; j < med_doc[i].length; j++)
				med_doc[i][j] = formula.get(j);
		}

		MultiLabelClinicalCases mcc = new MultiLabelClinicalCases(med_doc,
				doc_multilabel, label_set, new_vocab);

		return mcc;
	}

	public static List<String> getMedicines() throws IOException {
		List<String> medicines = new ArrayList<String>();

		File f = new File("src//file//vocab.txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));

		String line = "";

		while ((line = reader.readLine()) != null) {
			medicines.add(line);
		}
		medicines.remove("水");

		reader.close();

		return medicines;
	}

	public static Map<String, Set<String>> getLabelMedicines()
			throws IOException {

		/*
		 * 读取文件列表
		 */

		String path = "src//file//pre";
		result = new File("src//file//doclist(Pre_class).txt");
		out = new FileOutputStream(result, false);
		bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));
		readfile(path);
		bw.close();
		out.close();

		File f = new File("src//file//doclist(Pre_class).txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		Map<String, Set<String>> class_med_map = new HashMap<String, Set<String>>();

		List<String> med_vocab = getMedicines();

		while ((line = reader.readLine()) != null) {

			File f1 = new File(line); // 安神剂.txt
			BufferedReader reader_1 = new BufferedReader(new InputStreamReader(
					new FileInputStream(f1), "UTF-8"));
			String line_1 = "";

			StringBuilder sb = new StringBuilder();

			Set<String> medicines = new HashSet<String>();

			while ((line_1 = reader_1.readLine()) != null) {

				File f2 = new File(line_1); // root//安神剂//xxx.txt
				BufferedReader reader_2 = new BufferedReader(
						new InputStreamReader(new FileInputStream(f2), "UTF-8"));
				String line_2 = "";

				while ((line_2 = reader_2.readLine()) != null) {

					sb.append(line_2);

				}
				reader_2.close();

			}

			for (String med : med_vocab) {

				if (sb.toString().contains(med)) {
					medicines.add(med);
				}
			}

			reader_1.close();

			String function = line.substring(line.lastIndexOf("\\") + 1, line
					.indexOf("."));

			class_med_map.put(function, medicines);

		}
		reader.close();

		return class_med_map;
	}

}

class ClinicalCases {

	int[][] med_doc;

	String[] labels;

	public ClinicalCases(int[][] med_doc, String[] labels) {
		this.med_doc = med_doc;
		this.labels = labels;

	}

}

class MultiLabelClinicalCases {

	int[][] med_doc;
	List<List<String>> multilabels;

	List<String> label_set;

	List<String> vocab;

	public MultiLabelClinicalCases(int[][] med_doc,
			List<List<String>> multilabels, List<String> label_set,
			List<String> vocab) {

		this.med_doc = med_doc;
		this.multilabels = multilabels;
		this.label_set = label_set;
		this.vocab = vocab;

	}

}

class Corpus {

	int[][] med_doc;
	int[][] distinct_med_doc;
	int[] label_array;
	String[] map_keys;

	int[] distinct_label_array;

	Map<String, Set<Integer>> label_map;

	List<String> label_set;
	List<String> vocab;
	String[] distinct_map_keys;

	public Corpus(int[][] med_doc, int[] labels, List<String> label_set,
			List<String> vocab, String[] map_keys,
			Map<String, Set<Integer>> label_map, int[][] distinct_med_doc,
			List<String> distinct_map_keys, int[] distinct_label_array)
			throws IOException {
		this.med_doc = med_doc;
		this.label_array = labels;
		this.label_set = label_set;
		this.vocab = vocab;

		this.map_keys = map_keys;
		this.label_map = label_map;

		this.distinct_med_doc = distinct_med_doc;
		this.distinct_map_keys = new String[distinct_map_keys.size()];
		this.distinct_label_array = distinct_label_array;
		for (int i = 0; i < this.distinct_map_keys.length; i++)
			this.distinct_map_keys[i] = distinct_map_keys.get(i);

		/*
		 * 将vocab写入文件
		 */

		File file = new File("src//file//vocab.txt");
		OutputStream out = new FileOutputStream(file, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out,
				"utf-8"));

		for (String e : this.vocab) {
			bw.write(e);
			bw.newLine();
		}
		bw.close();
		out.close();

	}
}

// For zlabelLDA
class MedLabels {

	String[] medicines;
	String[] med_classes;

	public MedLabels(String[] medicines, String[] med_classes) {

		this.medicines = medicines;
		this.med_classes = med_classes;

	}

}

// For Constrained-LDA
class MedConstraint {

	Set<Integer> must_link; // 相辅相成
	Set<Integer> cannot_link;// 相畏相反

	public MedConstraint(Set<Integer> must, Set<Integer> cannot) {
		this.must_link = must;
		this.cannot_link = cannot;
	}

}
