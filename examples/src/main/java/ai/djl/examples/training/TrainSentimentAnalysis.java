package ai.djl.examples.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.TextDataset;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.Decoder;
import ai.djl.modality.nlp.Encoder;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.util.PairList;
import org.apache.commons.cli.ParseException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;

public class TrainSentimentAnalysis {
    public static void main(String[] args) throws IOException, ParseException {
        TrainSentimentAnalysis.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, ParseException {
        Arguments arguments = Arguments.parseArgs(args);

        try (Model model = Model.newInstance()) {
            // get training and validation dataset
            TextDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            TextDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            TrainableTextEmbedding sourceTextEmbedding = (TrainableTextEmbedding) trainingSet.getTextEmbedding(true);
            model.setBlock(getModel(sourceTextEmbedding));

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            config.addTrainingListeners(
                    TrainingListener.Defaults.logging(arguments.getOutputDir()));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), 10);
                Shape decoderInputShape = new Shape(arguments.getBatchSize(), 9);

                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape);

                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainingSet,
                        null,
                        arguments.getOutputDir(),
                        "trainSeqToSeq");

                TrainingResult result = trainer.getTrainingResult();
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

                Path modelSavePath = Paths.get(arguments.getOutputDir());
                System.out.println(modelSavePath.toAbsolutePath());
                model.save(modelSavePath, "trainSeqToSeq");

                return result;
            }
        }
    }

    private static Block getModel(TrainableTextEmbedding trainableTextEmbedding) {
        SequentialBlock sequentialBlock = new SequentialBlock();
        sequentialBlock
                .add(trainableTextEmbedding)
                .add(LSTM.builder().setNumStackedLayers(1).setSequenceLength(false).setStateSize(100).optStateOutput(false).build())
                .add(Linear.builder().setOutChannels(2).optFlatten(false).build());
        return sequentialBlock;
    }

    public static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        return new DefaultTrainingConfig(new MaskedSoftmaxCrossEntropyLoss())
                .optInitializer(new XavierInitializer())
                .optOptimizer(
                        Adam.builder()
                                .optLearningRateTracker(
                                        LearningRateTracker.fixedLearningRate(0.005f))
                                .build())
                .optDevices(Device.getDevices(arguments.getMaxGpus()));
    }

    public static TextDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                TatoebaEnglishFrenchDataset.builder()
                        .setValidLength(true)
                        .setSampling(arguments.getBatchSize(), true, true)
                        .optEmbeddingSize(32)
                        .optUsage(usage)
                        .optLimit(arguments.getLimit())
                        .optExecutor(Executors.newFixedThreadPool(arguments.getBatchSize()), arguments.getBatchSize())
                        .build();
        tatoebaEnglishFrenchDataset.prepare(new ProgressBar());
        return tatoebaEnglishFrenchDataset;
    }
}
