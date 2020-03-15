import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class MultithreadingRead {
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
     * 梯度下降法步长
     */
    private double stepLength;
    /**
     * 最大迭代次数
     */
    private int maxStep;
    /**
     * 权重矩阵初始化值
     */
    private double initWeight;
    /**
     * 训练后的权重矩阵
     */
    private double[] weights;

    /**
     * 每次迭代使用的样本数
     */
    private int batch_size;

    public double[] getWeights() {
        return weights;
    }
    // 训练数据
    private String trainFileName;
    // 测试数据
    private String testFileName;
    // 预测结果
    private String predictFileName;

    public MultithreadingRead (String trainFileName, String testFileName, String predictFileName) {
        this.trainFileName = trainFileName;
        this.testFileName = testFileName;
        this.predictFileName = predictFileName;

        this.stepLength = 0.034; //学习率
        this.maxStep = 800;
        this.initWeight = 1.0;
        this.batch_size = 200;
    }

    /*
    private void loadTrainingData() {
        double[][] matrix = loadFile(trainFileName, false);

        feature = new double[matrix.length][matrix[0].length];
        label = new int[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length - 1; j++) {
                feature[i][j] = matrix[i][j];
            }
            label[i] = (int) matrix[i][matrix[i].length - 1];
        }
    }*/

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
            feature[i][len - 1] = 0.0;
        }
    }

    private void initWeightMatrix() {
        int paraSize = feature[0].length;
        double[] weights = new double[paraSize];
        for (int i = 0; i < paraSize; i++) {
            weights[i] = initWeight;
        }
        this.weights = weights;
    }

    /**
     * 迭代标签值
     *
     * @return 预测标签值
     */
    private double[] getPredictLabel() {
        double[] predictLabels = new double[feature.length];
        for (int i = 0; i < predictLabels.length; i++) {
            double predictSum = 0;
            for (int j = 0; j < feature[i].length; j++) {
                predictSum += feature[i][j] * weights[j];
            }
            predictLabels[i] = sigmoid(predictSum);
        }
        return predictLabels;
    }

    /**
     * 计算权重矩阵偏差
     *
     * @return 权重矩阵偏差
     */
    /*
    private double[] getDeltaWeights() {
        double[] predictLabels = getPredictLabel();
        double[] deltaWeights = new double[feature[0].length];
        for (int i = 0; i < feature[0].length; i++) {
            deltaWeights[i] = 0;
            for (int j = 0; j < feature.length; j++) {
                deltaWeights[i] += feature[j][i] * (label[j] - predictLabels[j]);
            }
            deltaWeights[i] /= feature.length;
        }

        // System.out.println("Error: "+getErrorRate(predictLabels));

        return deltaWeights;
    }
    */

    // 计算误差率
    private double getErrorRate(double[] predictLabels) {
        double sumErr = 0.0;
        for (int i = 0; i < label.length; i++) {
            sumErr += Math.pow(this.label[i] - predictLabels[i], 2);
        }
        return sumErr;
    }

    /**
     * 迭代训练权重矩阵
     */

    //小批量梯度下降
    public void training() {
        //loadTrainingData(trainFileName, false);

        if (feature.length <= 0 || feature[0].length <= 0) {
            weights = null;
            return;
        }

        initWeightMatrix();
        int num = feature.length / batch_size; //每n个迭代遍历完所有样本一次

        for (int step = 0; step < maxStep; step++) {

            int batch_num = step % num * batch_size;
            double[] predictLabels = getPredictLabel();
            double[] deltaWeights = new double[feature[0].length];

            for(int j = 0; j < feature[0].length; j++) {
                deltaWeights[j] = 0;
                for(int k = batch_num; k < batch_num + batch_size; k++) {
                    deltaWeights[j] += feature[k][j] * (label[k] - predictLabels[k]);
                }
                deltaWeights[j] /= batch_size;
            }

            for (int j = 0; j < feature[0].length; j++) {
                weights[j] += stepLength * deltaWeights[j];
            }
        }

        this.feature = null;
        this.label = null;
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

    public void predict() {
        int[] predictLabel = new int[testFeature.length];
        for (int i = 0; i < testFeature.length; i++) {
            double sum = 0;
            for (int j = 0; j < testFeature[i].length; j++) {
                sum += testFeature[i][j] * weights[j];
            }
            predictLabel[i] = sigmoid(sum) > 0.5 ? 1 : 0;
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


    private double sigmoid(double x) {
        return 1.0d / (1.0d + Math.exp(-x));
    }


    // 命令行格式为test train_data.txt（训练集） test_data.txt(测试数据) predict.txt(预测结果) [debug]
    public static void main(String[] args) {
        BufferedReader reader = null;

        String trainFileName = "./data/train_data.txt";
        String testFileName = "./data/test_data.txt";
        String predictFileName = "./projects/student/result.txt";
        String answerFileName = "./projects/student/answer.txt";

        boolean isDebug = true;
        if (args.length >= 1) {
            if (args[0].equals("debug")) {
                isDebug = true;
            }
        }

        //1.初始化
        long start = System.currentTimeMillis();
        MultithreadingRead lr = new MultithreadingRead(trainFileName, testFileName, predictFileName);
        long end = System.currentTimeMillis();
        System.out.println("Init Time(s): " + (end - start) * 1.0 / 1000);

        //2.单线程读数据
        /*
        lr.loadTrainingData(trainFileName, false);
        lr.loadPredictingData(testFileName, false);
        System.out.println("Reading file Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();
        */

        //2.双线程读数据
        final CountDownLatch latch = new CountDownLatch(2);
        new Thread(new Runnable() {
            @Override
            public void run() {
                    lr.loadTrainingData(trainFileName, false);
                    latch.countDown();
            }
        }).start();

        new Thread(new Runnable() {
            @Override
            public void run() {
                    lr.loadPredictingData(testFileName, false);
                    latch.countDown();
            }
        }).start();

        System.out.println("Reading file Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();

        //3.训练
        try {
            latch.await();
            lr.training();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("Training Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();

        //3.训练
        /*
        lr.training();
        System.out.println("Training Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();
        */

        //4.预测
        lr.predict();
        System.out.println("Predict Time(s): " + (System.currentTimeMillis() - end) * 1.0 / 1000);
        end = System.currentTimeMillis();
        if (isDebug) {

            double[][] matrix = lr.loadFile(predictFileName, false);
            int[] predict = new int[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                predict[i] = (int) matrix[i][0];
            }


            matrix = lr.loadFile(answerFileName, false);
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
