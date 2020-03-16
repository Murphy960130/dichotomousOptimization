import sun.jvm.hotspot.utilities.GenericArray;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NativeBayes {
    /**
     * 特征矩阵
     */
    private double[][] feature;
    /**
     * 样本标签
     */
    private int[] label;

    /**
     * 测试数据样本
     */
    private double[][] testFeature;

    /**
     * 先验概率P(y0), P(y1)
     */
    private double[] P;

    /**
     * 每一个特征的均值
     */
    private double[][] mean;

    /**
     * 每一列特征的标准差
     */
    private double[][] ssd;

    /**
     * 特征值允许的误差
     */
    double eplision = 1e-6;

    // 训练数据
    private String trainFileName;
    // 测试数据
    private String testFileName;
    // 预测结果
    private String predictFileName;

    public NativeBayes (String trainFileName, String testFileName, String predictFileName) {
        this.trainFileName = trainFileName;
        this.testFileName = testFileName;
        this.predictFileName = predictFileName;
    }

    //加载训练样本
    private void loadTrainingData(String fileName, boolean skipTitle) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException exception) {
            System.err.println(fileName + " File Not Found");
        }
        List<List<Double>> listArr = new ArrayList<>();
        String line = "";
        try {
            if (skipTitle) {
                reader.readLine();
            }
            while ((line = reader.readLine()) != null) {
                List<Double> list = new ArrayList<>();
                String item[] = line.split(",");
                for (int i = 0; i < item.length; i++) {
                    list.add(Double.parseDouble(item[i]));
                }
                listArr.add(list);
            }
        } catch (IOException exception) {
            System.err.println(exception.getMessage());
        }

        feature = new double[listArr.size()][listArr.get(0).size()];
        label = new int[listArr.size()];
        for (int i = 0; i < listArr.size(); i++) {
            int len = listArr.get(i).size();
            for (int j = 0; j < len; j++) {
                feature[i][j] = listArr.get(i).get(j);
            }
            label[i] = (int) feature[i][len - 1];
            feature[i][len - 1] = 0.0f;
        }
    }

    //加载测试样本
    private void loadPredictingData(String fileName, boolean skipTitle) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException exception) {
            System.err.println(fileName + " File Not Found");
        }
        List<List<Double>> listArr = new ArrayList<>();
        String line = "";
        try {
            if (skipTitle) {
                reader.readLine();
            }
            while ((line = reader.readLine()) != null) {
                List<Double> list = new ArrayList<>();
                String item[] = line.split(",");
                for (int i = 0; i < item.length; i++) {
                    list.add(Double.parseDouble(item[i]));
                }
                listArr.add(list);
            }
        } catch (IOException exception) {
            System.err.println(exception.getMessage());
        }

        testFeature = new double[listArr.size()][listArr.get(0).size()];
        for (int i = 0; i < listArr.size(); i++) {
            for (int j = 0; j < listArr.get(i).size(); j++) {
                testFeature[i][j] = listArr.get(i).get(j);
            }
        }
    }

    public double[][] loadFile(String fileName, boolean skipTitle) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException exception) {
            System.err.println(fileName + " File Not Found");
            return null;
        }
        List<List<Double>> listArr = new ArrayList<>();
        String line = "";
        try {
            if (skipTitle) {
                reader.readLine();
            }
            while ((line = reader.readLine()) != null) {
                List<Double> list = new ArrayList<>();
                String item[] = line.split(",");
                for (int i = 0; i < item.length; i++) {
                    list.add(Double.parseDouble(item[i]));
                }
                listArr.add(list);
            }
        } catch (IOException exception) {
            System.err.println(exception.getMessage());
        }

        double[][] matrix = new double[listArr.size()][listArr.get(0).size()];
        for (int i = 0; i < listArr.size(); i++) {
            for (int j = 0; j < listArr.get(i).size(); j++) {
                matrix[i][j] = listArr.get(i).get(j);
            }
        }
        return matrix;
    }

    //训练。统计y0和y1均值，方差，先验概率
    private void training() {
        P =  new double[2];
        mean = new double[2][feature[0].length - 1];
        ssd = new double[2][feature[0].length - 1];

        int cnt_0 = 0;
        for(int i = 0; i < label.length; i++) {
            if(label[i] == 0) cnt_0++;
        }

        //先验概率
        P[0] = (double) cnt_0 / label.length;
        P[1] = 1 - P[0];

        //均值
        for(int j = 0; j < feature[0].length - 1; j++) {
            for(int i = 0; i < feature.length; i++) {
                mean[label[i]][j] += feature[i][j];
            }
            mean[0][j] /= cnt_0;
            mean[1][j] /= (label.length - cnt_0);
        }

        //标准差
        for(int j = 0; j < feature[0].length - 1; j++) {
            for(int i = 0; i < feature.length; i++) {
                ssd[label[i]][j] += Math.pow(feature[i][j] - mean[label[i]][j], 2);
            }
            ssd[0][j] = Math.sqrt(ssd[0][j] / cnt_0);
            ssd[1][j] = Math.sqrt(ssd[1][j] / (label.length - cnt_0));
        }
    }


    private double gaussian(double mean, double ssd, double x) {
        double sqrt2pi = Math.sqrt(2 * Math.PI);
        double ePart = Math.exp(- Math.pow(x - mean, 2) / (2 * Math.pow(ssd, 2)));
        return ePart / (sqrt2pi * ssd);

    }

    private int predictOneSample(double[] sample) {
        double P_0 = 1.0;
        double P_1 = 1.0;

        for(int j = 0; j < sample.length; j++) {
            P_0 *= gaussian(mean[0][j], ssd[0][j], sample[j]);
            P_1 *= gaussian(mean[1][j], ssd[1][j], sample[j]);
        }

        P_0 *= P[0];
        P_1 *= P[1];

        return P_0 > P_1 ? 0 : 1;
    }

    public void predict() {
        int[] predictLabel = new int[testFeature.length];

        for(int i = 0; i < testFeature.length; i++) {
            predictLabel[i] = predictOneSample(testFeature[i]);
        }
        savePredictResult(predictLabel);
    }


    private void savePredictResult(int[] predictLabel) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(predictFileName));
            for (int i = 0; i < predictLabel.length; i++) {
                out.write(predictLabel[i] + "\n");
            }
            out.close();
        } catch (IOException exception) {
            System.err.println(exception.getMessage());
        }
    }


    // 命令行格式为test train_data.txt（训练集） test_data.txt(测试数据) predict.txt(预测结果) [debug]
    public static void main(String[] args) {
        BufferedReader reader = null;

        String trainFileName = "/data/train_data.txt";
        String testFileName = "/data/test_data.txt";
        String predictFileName = "/projects/student/result.txt";
        String answerFileName = "/projects/student/answer.txt";

        boolean isDebug = false;
        if (args.length >= 1) {
            if (args[0].equals("debug")) {
                isDebug = true;
            }
        }

        //1.初始化
        long start = System.currentTimeMillis();
        NativeBayes nb = new NativeBayes(trainFileName, testFileName, predictFileName);
        long end = System.currentTimeMillis();
        System.out.println("Init Time(s): " + (end - start) * 1.0 / 1000);

        //2.单线程读数据
        nb.loadTrainingData(trainFileName, false);
        nb.loadPredictingData(testFileName, false);
        System.out.println("Reading file Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();


        //3.训练
        nb.training();
        System.out.println("Training Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();

        //4.预测
        nb.predict();
        System.out.println("Predict Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();
        if (isDebug) {

            double[][] matrix = nb.loadFile(predictFileName, false);
            int[] predict = new int[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                predict[i] = (int) matrix[i][0];
            }


            matrix = nb.loadFile(answerFileName, false);
            int[] answer = new int[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                answer[i] = (int) matrix[i][0];
            }

            int accCount = 0;
            for (int i = 0; i < predict.length; i++) {
                if (predict[i] == answer[i]) {
                    accCount++;
                }
            }
            System.out.println("Right Count:" + accCount + "/" + predict.length);
            System.out.println("Accuracy:" + (1.0f * accCount / predict.length));
            System.out.println("Mark Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        }
    }
}



