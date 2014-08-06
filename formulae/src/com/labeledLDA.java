package com;

/*
 * Labeled-LDA (In EMNLP'09)
 * @author : yaoliang
 * 
 */

public class labeledLDA {

	int[][] documents;

	int V;

	int K; // 主题数，等于标签集合的大小

	double[][] alpha;

	double beta;

	int[][] z;

	int[][] nw;

	int[][] nd;

	int[] nwsum;

	int[] ndsum;

	int iterations;

	public labeledLDA(int[][] docs, int V) {
		this.documents = docs;
		this.V = V;

	}

	public void initialState() {

		int D = documents.length;
		nw = new int[V][K];
		nd = new int[D][K];
		nwsum = new int[K];
		ndsum = new int[D];

		z = new int[D][];
		for (int d = 0; d < D; d++) {

			int Nd = documents[d].length;

			z[d] = new int[Nd];

			for (int n = 0; n < Nd; n++) {
				int topic = (int) (Math.random() * K);

				z[d][n] = topic;

				nw[documents[d][n]][topic]++;

				nd[d][topic]++;

				nwsum[topic]++;
			}
			ndsum[d] = Nd;
		}

	}

	public void markovChain(int K, double[][] alpha, double beta, int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.iterations = iterations;

		initialState();

		for (int i = 0; i < this.iterations; i++) {
			// System.out.println("iterations: "+i);
			gibbs();
		}
	}

	public void gibbs() {

		for (int d = 0; d < z.length; d++) {
			for (int n = 0; n < z[d].length; n++) {

				int topic = sampleFullConditional(d, n);
				z[d][n] = topic;

			}
		}
	}

	int sampleFullConditional(int d, int n) {

		int topic = z[d][n];
		nw[documents[d][n]][topic]--;
		nd[d][topic]--;
		nwsum[topic]--;
		ndsum[d]--;

		double[] p = new double[K];

		double alpha_sum = 0.0;

		for (int i = 0; i < K; i++)
			alpha_sum += alpha[d][i];

		for (int k = 0; k < K; k++) {

			p[k] = (nd[d][k] + alpha[d][k]) / (ndsum[d] + alpha_sum)
					* (nw[documents[d][n]][k] + beta) / (nwsum[k] + V * beta);
		}
		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1];
		for (int t = 0; t < K; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}
		nw[documents[d][n]][topic]++;
		nd[d][topic]++;
		nwsum[topic]++;
		ndsum[d]++;
		return topic;

	}

	public double[][] estimateTheta() {
		double[][] theta = new double[documents.length][K];
		for (int d = 0; d < documents.length; d++) {
			double alpha_sum = 0.0;
			for (int k = 0; k < K; k++)
				alpha_sum += alpha[d][k];

			for (int k = 0; k < K; k++) {
				theta[d][k] = (nd[d][k] + alpha[d][k]) / (ndsum[d] + alpha_sum);
			}
		}
		return theta;
	}

	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		return phi;
	}

	// predict label (topic) using naive bayes
	public double[] predict(int[] doc, double[][] phi) {
		double[] prob = new double[K];

		for (int k = 0; k < K; k++) {
			double product = 1.0;
			for (int i = 0; i < doc.length; i++) {
				product *= phi[k][doc[i]];
			}
			prob[k] = product;
		}
		return prob;

	}

	public double[] predict(int[] doc) {

		double[] prob = new double[K];

		int[] count = new int[K];

		for (int k = 0; k < K; k++) {

			int sum = 0;
			for (int l = 0; l < doc.length; l++) {

				sum += nw[doc[l]][k];

			}
			count[k] = sum;

		}
		int sum = 0;

		for (int k = 0; k < K; k++) {

			sum += count[k];

		}

		for (int k = 0; k < K; k++) {

			if (doc.length > 0)

				prob[k] = (double) count[k] / sum * doc.length + (double) 50
						/ K;
			else
				prob[k] = (double) 50 / K;
		}

		return prob;
	}

	public static void main(String args[]) {

		int[][] documents = {
				{ 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6 },
				{ 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2 },
				{ 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0 },
				{ 5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0 },
				{ 2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0 },
				{ 5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2 },

		};
		int[][] doclabels = { { 1, 2 }, { 2, 3 }, { 4 }, { 2 }, { 0 },
				{ 5, 2 }, };
		int V = 7;
		int K = 6;// 标签集合的大小
		int iterations = 1000;
		labeledLDA l = new labeledLDA(documents, V);
		double[][] alpha = new double[documents.length][K];
		for (int i = 0; i < doclabels.length; i++) {
			for (int j = 0; j < doclabels[i].length; j++) {
				alpha[i][doclabels[i][j]] = (double) 50 / K;
			}
		}

		double beta = 0.1;
		l.markovChain(K, alpha, beta, iterations);
		double[][] theta = l.estimateTheta();
		double[][] phi = l.estimatePhi();

		System.out.println("Document Topic association:");

		for (int i = 0; i < theta.length; i++) {
			for (int j = 0; j < theta[i].length; j++)
				System.out.print(theta[i][j] + " ");
			System.out.println();
		}

		System.out.println("Topic word association:");

		for (int i = 0; i < phi.length; i++) {
			for (int j = 0; j < phi[i].length; j++)
				System.out.print(phi[i][j] + " ");
			System.out.println();
		}

		int[] test_doc = { 5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0 };
		double[] predict_result = l.predict(test_doc, phi);// predict
		for (int i = 0; i < K; i++)
			System.out.println(predict_result[i]);

		System.out.println(Common.perplexity(theta, phi, documents));
	}

}
